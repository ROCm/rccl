/**
 * Device Linker - Merges specialized kernel device objects into a single ELF.
 * 
 * Usage:
 *   device_linker -o output.o --dispatcher minimal_device.o --host-table host_table.cpp input1.o input2.o ...
 *   device_linker -o output.o --dispatcher minimal_device.o --host-table host_table.cpp --input-dir <dir>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <elf.h>

namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

constexpr int FUNC_COUNT = 859;  // Max funcId is 858
constexpr int FUNC_ALIGNMENT = 256;

// ============================================================================
// Memory-mapped file wrapper
// ============================================================================

class MappedFile {
public:
    MappedFile(const std::string& path) : fd_(-1), data_(nullptr), size_(0) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            fprintf(stderr, "Error: Cannot open %s\n", path.c_str());
            return;
        }
        
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            fd_ = -1;
            return;
        }
        
        size_ = st.st_size;
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            close(fd_);
            fd_ = -1;
        }
    }
    
    ~MappedFile() {
        if (data_) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
    }
    
    // Non-copyable, non-movable (owns raw resources)
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&) = delete;
    MappedFile& operator=(MappedFile&&) = delete;
    
    bool valid() const { return data_ != nullptr; }
    size_t size() const { return size_; }
    const void* data() const { return data_; }
    
    template<typename T>
    const T* at(size_t offset) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data_) + offset);
    }
    
    const char* str(size_t offset) const {
        return static_cast<const char*>(data_) + offset;
    }

private:
    int fd_;
    void* data_;
    size_t size_;
};

// ============================================================================
// ELF utilities
// ============================================================================

struct Section {
    std::string name;
    uint32_t type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
};

class ElfFile {
public:
    ElfFile(const MappedFile& file) : file_(file) {
        if (!file.valid()) return;
        
        ehdr_ = file.at<Elf64_Ehdr>(0);
        if (memcmp(ehdr_->e_ident, ELFMAG, SELFMAG) != 0) {
            ehdr_ = nullptr;
            return;
        }
        
        // Parse section headers
        const Elf64_Shdr* shdrs = file.at<Elf64_Shdr>(ehdr_->e_shoff);
        const Elf64_Shdr& shstrtab = shdrs[ehdr_->e_shstrndx];
        const char* strtab = file.str(shstrtab.sh_offset);
        
        for (int i = 0; i < ehdr_->e_shnum; i++) {
            Section sec;
            sec.name = strtab + shdrs[i].sh_name;
            sec.type = shdrs[i].sh_type;
            sec.flags = shdrs[i].sh_flags;
            sec.addr = shdrs[i].sh_addr;
            sec.offset = shdrs[i].sh_offset;
            sec.size = shdrs[i].sh_size;
            sections_.push_back(sec);
        }
    }
    
    bool valid() const { return ehdr_ != nullptr; }
    const Elf64_Ehdr* ehdr() const { return ehdr_; }
    const std::vector<Section>& sections() const { return sections_; }
    
    const Section* findSection(const std::string& name) const {
        for (const auto& sec : sections_) {
            if (sec.name == name) return &sec;
        }
        return nullptr;
    }
    
    const void* sectionData(const Section& sec) const {
        return file_.at<void>(sec.offset);
    }
    
    std::vector<uint8_t> getSectionBytes(const Section& sec) const {
        const uint8_t* data = file_.at<uint8_t>(sec.offset);
        return std::vector<uint8_t>(data, data + sec.size);
    }

private:
    const MappedFile& file_;
    const Elf64_Ehdr* ehdr_ = nullptr;
    std::vector<Section> sections_;
};

// ============================================================================
// Kernel info extracted from each .device.o
// ============================================================================

struct KernelInfo {
    std::string source_file;
    std::string mangled_name;
    uint64_t func_offset;      // Offset within .text
    uint64_t func_size;
    std::vector<uint8_t> code; // The actual machine code
    
    // Resource requirements
    int vgpr_count = 0;
    int sgpr_count = 0;
    int lds_size = 0;
    int stack_size = 0;
};

// Parse symbols from .symtab
std::string findDevFunc(const MappedFile& file, const ElfFile& elf, 
                        uint64_t& offset, uint64_t& size) {
    const Section* symtab = elf.findSection(".symtab");
    const Section* strtab = elf.findSection(".strtab");
    if (!symtab || !strtab) return "";
    
    const char* strings = file.str(strtab->offset);
    const Elf64_Sym* syms = file.at<Elf64_Sym>(symtab->offset);
    size_t nsyms = symtab->size / sizeof(Elf64_Sym);
    
    for (size_t i = 0; i < nsyms; i++) {
        if (ELF64_ST_TYPE(syms[i].st_info) == STT_FUNC) {
            const char* name = strings + syms[i].st_name;
            if (strstr(name, "ncclDevFunc_") != nullptr) {
                offset = syms[i].st_value;
                size = syms[i].st_size;
                return name;
            }
        }
    }
    return "";
}

// Parse resource requirements from .note section
void parseNoteMetadata(const MappedFile& file, const ElfFile& elf, KernelInfo& info) {
    const Section* note = elf.findSection(".note");
    if (!note) return;
    
    // Walk note entries
    size_t pos = 0;
    while (pos < note->size) {
        const uint32_t* hdr = file.at<uint32_t>(note->offset + pos);
        uint32_t namesz = hdr[0];
        uint32_t descsz = hdr[1];
        uint32_t type = hdr[2];
        
        size_t name_off = pos + 12;
        size_t desc_off = ((name_off + namesz + 3) & ~3);
        
        // Look for AMDGPU metadata (type 32)
        if (type == 32) {
            const char* desc = file.str(note->offset + desc_off);
            // Parse MessagePack metadata - look for known keys
            // This is a simplified parser that looks for the string keys
            std::string_view data(desc, descsz);
            
            auto findInt = [&](const char* key) -> int {
                auto pos = data.find(key);
                if (pos == std::string_view::npos) return 0;
                pos += strlen(key);
                if (pos >= data.size()) return 0;
                uint8_t b = data[pos];
                if (b <= 0x7f) return b;  // fixint
                if (b == 0xcc && pos + 1 < data.size()) return (uint8_t)data[pos + 1];  // uint8
                if (b == 0xcd && pos + 2 < data.size()) return ((uint8_t)data[pos + 1] << 8) | (uint8_t)data[pos + 2];  // uint16
                if (b == 0xce && pos + 4 < data.size()) {
                    return ((uint8_t)data[pos + 1] << 24) | ((uint8_t)data[pos + 2] << 16) |
                           ((uint8_t)data[pos + 3] << 8) | (uint8_t)data[pos + 4];  // uint32
                }
                return 0;
            };
            
            info.vgpr_count = findInt(".vgpr_count");
            info.sgpr_count = findInt(".sgpr_count");
            info.lds_size = findInt(".group_segment_fixed_size");
            info.stack_size = findInt(".private_segment_fixed_size");
        }
        
        pos = ((desc_off + descsz + 3) & ~3);
    }
}

// Process a single device object file
KernelInfo processDeviceObject(const std::string& path) {
    KernelInfo info;
    info.source_file = fs::path(path).filename().string();
    
    MappedFile file(path);
    if (!file.valid()) return info;
    
    ElfFile elf(file);
    if (!elf.valid()) return info;
    
    // Find ncclDevFunc_* symbol
    uint64_t func_offset, func_size;
    info.mangled_name = findDevFunc(file, elf, func_offset, func_size);
    if (info.mangled_name.empty()) return info;
    
    info.func_offset = func_offset;
    info.func_size = func_size;
    
    // Get .text section and extract function code
    const Section* text = elf.findSection(".text");
    if (!text) return info;
    
    // Calculate file offset of function
    uint64_t file_offset = text->offset + (func_offset - text->addr);
    const uint8_t* code_ptr = file.at<uint8_t>(file_offset);
    info.code.assign(code_ptr, code_ptr + func_size);
    
    // Parse metadata
    parseNoteMetadata(file, elf, info);
    
    return info;
}

// ============================================================================
// FuncId mapping from host_table.cpp
// ============================================================================

std::unordered_map<std::string, int> parseHostTable(const std::string& path) {
    std::unordered_map<std::string, int> mapping;
    
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Warning: Cannot open host_table.cpp: %s\n", path.c_str());
        return mapping;
    }
    
    // Match: {key, id}, // Comment COLL ALGO PROTO REDOP TYPE ACC PIPELINE
    std::regex pattern(R"(\{(\d+),\s*(\d+)\},\s*//\s*(.+))");
    std::string line;
    
    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, pattern)) {
            int func_id = std::stoi(match[2]);
            std::string comment = match[3];
            
            // Parse comment: "AllReduce RING LL Sum f32 0 0 2"
            std::istringstream iss(comment);
            std::vector<std::string> parts;
            std::string part;
            while (iss >> part) parts.push_back(part);
            
            if (parts.size() >= 7) {
                // Build lookup key: Coll_Algo_Proto_Redop_Type_Acc_Pipeline
                std::string key = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" +
                                  parts[3] + "_" + parts[4] + "_" + parts[5] + "_" + parts[6];
                mapping[key] = func_id;
            }
        }
    }
    
    return mapping;
}

// Demangle to extract function name components
std::string demangleToKey(const std::string& mangled) {
    // _Z48ncclDevFunc_AllReduce_RING_LL_Sum_f32_0_0_2v
    // -> AllReduce_RING_LL_Sum_f32_0_0
    
    if (mangled.substr(0, 2) != "_Z") return "";
    
    size_t i = 2;
    while (i < mangled.size() && isdigit(mangled[i])) i++;
    if (i == 2) return "";
    
    int len = std::stoi(mangled.substr(2, i - 2));
    std::string name = mangled.substr(i, len);
    
    // Remove "ncclDevFunc_" prefix
    const char* prefix = "ncclDevFunc_";
    if (name.substr(0, strlen(prefix)) == prefix) {
        name = name.substr(strlen(prefix));
    }
    
    // Parse: AllReduce_RING_LL_Sum_f32_0_0_2
    // Remove trailing unroll number for lookup
    auto parts = std::vector<std::string>();
    std::istringstream iss(name);
    std::string part;
    while (std::getline(iss, part, '_')) parts.push_back(part);
    
    if (parts.size() < 8) return "";
    
    // Rebuild without the last part (unroll)
    std::string key;
    for (size_t j = 0; j < parts.size() - 1; j++) {
        if (j > 0) key += "_";
        key += parts[j];
    }
    
    return key;
}

// ============================================================================
// ELF Builder
// ============================================================================

class ElfBuilder {
public:
    void setFlags(uint32_t flags) { flags_ = flags; }
    
    void addSection(const std::string& name, uint32_t type, uint64_t flags,
                    uint64_t addr, const std::vector<uint8_t>& data, 
                    uint64_t align = 1, uint64_t entsize = 0) {
        SectionDef sec;
        sec.name = name;
        sec.type = type;
        sec.flags = flags;
        sec.addr = addr;
        sec.data = data;
        sec.align = align;
        sec.entsize = entsize;
        sections_.push_back(std::move(sec));
    }
    
    std::vector<uint8_t> build() {
        // Build section name string table
        std::vector<uint8_t> shstrtab;
        shstrtab.push_back(0);  // Empty string at index 0
        std::vector<uint32_t> name_offsets;
        name_offsets.push_back(0);  // NULL section
        
        for (const auto& sec : sections_) {
            name_offsets.push_back(shstrtab.size());
            shstrtab.insert(shstrtab.end(), sec.name.begin(), sec.name.end());
            shstrtab.push_back(0);
        }
        // Add .shstrtab name
        uint32_t shstrtab_name_off = shstrtab.size();
        const char* shstrtab_name = ".shstrtab";
        shstrtab.insert(shstrtab.end(), shstrtab_name, shstrtab_name + strlen(shstrtab_name) + 1);
        
        // Calculate layout
        size_t offset = 64;  // ELF header
        std::vector<size_t> section_offsets;
        
        for (const auto& sec : sections_) {
            offset = (offset + sec.align - 1) & ~(sec.align - 1);
            section_offsets.push_back(offset);
            offset += sec.data.size();
        }
        
        // shstrtab section (no alignment needed, align=1)
        size_t shstrtab_offset = offset;
        offset += shstrtab.size();
        
        // Section headers
        offset = (offset + 8 - 1) & ~(8 - 1);
        size_t shdr_offset = offset;
        size_t num_sections = sections_.size() + 2;  // NULL + sections + shstrtab
        
        // Build output
        std::vector<uint8_t> out(shdr_offset + num_sections * 64);
        
        // ELF header
        Elf64_Ehdr ehdr = {};
        memcpy(ehdr.e_ident, ELFMAG, SELFMAG);
        ehdr.e_ident[EI_CLASS] = ELFCLASS64;
        ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
        ehdr.e_ident[EI_VERSION] = EV_CURRENT;
        ehdr.e_ident[EI_OSABI] = 64;  // ELFOSABI_AMDGPU_HSA
        ehdr.e_type = ET_REL;
        ehdr.e_machine = 224;  // EM_AMDGPU
        ehdr.e_version = EV_CURRENT;
        ehdr.e_shoff = shdr_offset;
        ehdr.e_flags = flags_;
        ehdr.e_ehsize = 64;
        ehdr.e_shentsize = 64;
        ehdr.e_shnum = num_sections;
        ehdr.e_shstrndx = num_sections - 1;
        memcpy(out.data(), &ehdr, sizeof(ehdr));
        
        // Section data
        for (size_t i = 0; i < sections_.size(); i++) {
            memcpy(out.data() + section_offsets[i], 
                   sections_[i].data.data(), sections_[i].data.size());
        }
        memcpy(out.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());
        
        // Section headers
        auto writeShdr = [&](size_t idx, uint32_t name, uint32_t type, uint64_t flags,
                            uint64_t addr, uint64_t offset, uint64_t size,
                            uint32_t link, uint32_t info, uint64_t align, uint64_t entsize) {
            Elf64_Shdr shdr = {};
            shdr.sh_name = name;
            shdr.sh_type = type;
            shdr.sh_flags = flags;
            shdr.sh_addr = addr;
            shdr.sh_offset = offset;
            shdr.sh_size = size;
            shdr.sh_link = link;
            shdr.sh_info = info;
            shdr.sh_addralign = align;
            shdr.sh_entsize = entsize;
            memcpy(out.data() + shdr_offset + idx * 64, &shdr, sizeof(shdr));
        };
        
        // NULL section
        writeShdr(0, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);
        
        // User sections
        for (size_t i = 0; i < sections_.size(); i++) {
            const auto& sec = sections_[i];
            writeShdr(i + 1, name_offsets[i + 1], sec.type, sec.flags,
                      sec.addr, section_offsets[i], sec.data.size(),
                      0, 0, sec.align, sec.entsize);
        }
        
        // .shstrtab
        writeShdr(num_sections - 1, shstrtab_name_off, SHT_STRTAB, 0,
                  0, shstrtab_offset, shstrtab.size(), 0, 0, 1, 0);
        
        return out;
    }

private:
    struct SectionDef {
        std::string name;
        uint32_t type;
        uint64_t flags;
        uint64_t addr;
        std::vector<uint8_t> data;
        uint64_t align;
        uint64_t entsize;
    };
    
    uint32_t flags_ = 0;
    std::vector<SectionDef> sections_;
};

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* prog) {
    fprintf(stderr, "Usage: %s -o output.o --dispatcher disp.o --host-table host_table.cpp [--input-dir dir | files...]\n", prog);
}

int main(int argc, char** argv) {
    std::string output_path;
    std::string dispatcher_path;
    std::string host_table_path;
    std::string input_dir;
    std::vector<std::string> input_files;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--dispatcher" && i + 1 < argc) {
            dispatcher_path = argv[++i];
        } else if (arg == "--host-table" && i + 1 < argc) {
            host_table_path = argv[++i];
        } else if (arg == "--input-dir" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg[0] != '-') {
            input_files.push_back(arg);
        }
    }
    
    if (output_path.empty() || dispatcher_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Collect input files from directory if specified
    if (!input_dir.empty()) {
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.path().extension() == ".o" && 
                entry.path().string().find(".device.o") != std::string::npos) {
                input_files.push_back(entry.path().string());
            }
        }
        std::sort(input_files.begin(), input_files.end());
    }
    
    if (input_files.empty()) {
        fprintf(stderr, "Error: No input files\n");
        return 1;
    }
    
    printf("Device Linker: Processing %zu input files\n", input_files.size());
    
    // Parse host_table for funcId mapping
    auto funcid_map = parseHostTable(host_table_path);
    printf("Loaded %zu funcId mappings from host_table.cpp\n", funcid_map.size());
    
    // Process input files in parallel
    std::vector<KernelInfo> kernels(input_files.size());
    std::mutex progress_mutex;
    int processed = 0;
    
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            kernels[i] = processDeviceObject(input_files[i]);
            
            std::lock_guard<std::mutex> lock(progress_mutex);
            processed++;
            if (processed % 100 == 0) {
                printf("  Processed %d/%zu...\n", processed, input_files.size());
            }
        }
    };
    
    // Use thread pool
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    size_t chunk_size = (input_files.size() + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, input_files.size());
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    // Find max resource requirements
    int max_vgpr = 0, max_sgpr = 0, max_lds = 0, max_stack = 0;
    for (const auto& k : kernels) {
        max_vgpr = std::max(max_vgpr, k.vgpr_count);
        max_sgpr = std::max(max_sgpr, k.sgpr_count);
        max_lds = std::max(max_lds, k.lds_size);
        max_stack = std::max(max_stack, k.stack_size);
    }
    printf("Max resources: VGPR=%d, SGPR=%d, LDS=%d, Stack=%d\n", 
           max_vgpr, max_sgpr, max_lds, max_stack);
    
    // Load dispatcher
    MappedFile disp_file(dispatcher_path);
    if (!disp_file.valid()) {
        fprintf(stderr, "Error: Cannot load dispatcher: %s\n", dispatcher_path.c_str());
        return 1;
    }
    ElfFile dispatcher(disp_file);
    if (!dispatcher.valid()) {
        fprintf(stderr, "Error: Invalid dispatcher ELF\n");
        return 1;
    }
    
    // Get dispatcher sections
    const Section* disp_note = dispatcher.findSection(".note");
    const Section* disp_rodata = dispatcher.findSection(".rodata");
    const Section* disp_text = dispatcher.findSection(".text");
    const Section* disp_bss = dispatcher.findSection(".bss");
    
    if (!disp_text || !disp_bss) {
        fprintf(stderr, "Error: Dispatcher missing required sections\n");
        return 1;
    }
    
    // Build merged code section
    std::vector<uint8_t> disp_text_data = dispatcher.getSectionBytes(*disp_text);
    
    // Align to 256 for function code
    while (disp_text_data.size() % FUNC_ALIGNMENT != 0) {
        disp_text_data.push_back(0);
    }
    
    uint64_t func_code_vaddr = disp_text->addr + disp_text_data.size();
    printf("Function code starts at vaddr 0x%lx\n", func_code_vaddr);
    
    // Build function table and append code
    std::vector<uint64_t> func_table(FUNC_COUNT, 0);
    int mapped_count = 0;
    size_t total_code_size = 0;
    
    std::vector<std::string> unmapped;
    int empty_name = 0, empty_code = 0, out_of_range = 0;
    for (const auto& k : kernels) {
        if (k.mangled_name.empty()) { empty_name++; continue; }
        if (k.code.empty()) { empty_code++; continue; }
        
        std::string key = demangleToKey(k.mangled_name);
        auto it = funcid_map.find(key);
        if (it == funcid_map.end()) {
            unmapped.push_back(k.mangled_name + " -> key='" + key + "'");
            continue;
        }
        
        int funcid = it->second;
        if (funcid < 0 || funcid >= FUNC_COUNT) {
            out_of_range++;
            continue;
        }
        
        // Record function address (current end of .text section)
        uint64_t func_vaddr = disp_text->addr + disp_text_data.size();
        
        func_table[funcid] = func_vaddr;
        
        // Append code
        disp_text_data.insert(disp_text_data.end(), k.code.begin(), k.code.end());
        total_code_size += k.code.size();
        
        // Align for next function
        while (disp_text_data.size() % FUNC_ALIGNMENT != 0) {
            disp_text_data.push_back(0);
        }
        
        mapped_count++;
    }
    
    printf("Mapped %d functions, total code size: %zu bytes\n", mapped_count, total_code_size);
    printf("Skipped: %d empty name, %d empty code, %d out of range funcId\n", empty_name, empty_code, out_of_range);
    fflush(stdout);
    
    if (!unmapped.empty()) {
        printf("Unmapped functions (%zu):\n", unmapped.size());
        for (size_t i = 0; i < std::min(unmapped.size(), size_t(10)); i++) {
            printf("  %s\n", unmapped[i].c_str());
        }
        if (unmapped.size() > 10) {
            printf("  ... and %zu more\n", unmapped.size() - 10);
        }
    }
    
    // Build .data section (function tables)
    // Layout: table_1[FUNC_COUNT], table_2[FUNC_COUNT], table_4[FUNC_COUNT]
    size_t table_size = FUNC_COUNT * 8;
    std::vector<uint8_t> data_section(table_size * 3, 0);
    
    // Populate table_2 (middle table, for unroll=2)
    size_t table_2_offset = table_size;
    for (int i = 0; i < FUNC_COUNT; i++) {
        memcpy(data_section.data() + table_2_offset + i * 8, &func_table[i], 8);
    }
    
    // Build output ELF
    ElfBuilder builder;
    builder.setFlags(dispatcher.ehdr()->e_flags);
    
    // .note
    if (disp_note) {
        auto note_data = dispatcher.getSectionBytes(*disp_note);
        builder.addSection(".note", SHT_NOTE, SHF_ALLOC, disp_note->addr, note_data, 4);
    }
    
    // .rodata
    if (disp_rodata) {
        auto rodata_data = dispatcher.getSectionBytes(*disp_rodata);
        
        // Update kernel descriptors with max resources
        // KD offsets: 0, 64, 128 (3 kernels)
        auto updateKD = [&](size_t off) {
            if (off + 64 > rodata_data.size()) return;
            
            // LDS (offset 0)
            uint32_t lds = max_lds;
            memcpy(rodata_data.data() + off, &lds, 4);
            
            // Stack (offset 4)
            uint32_t stack = max_stack;
            memcpy(rodata_data.data() + off + 4, &stack, 4);
            
            // RSRC1 (offset 0x30) - VGPR/SGPR encoding
            uint32_t rsrc1;
            memcpy(&rsrc1, rodata_data.data() + off + 0x30, 4);
            int vgpr_granule = (max_vgpr + 3) / 4 - 1;
            int sgpr_granule = (max_sgpr + 7) / 8 - 1;
            rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_granule & 0x3F) | ((sgpr_granule & 0xF) << 6);
            memcpy(rodata_data.data() + off + 0x30, &rsrc1, 4);
        };
        
        updateKD(0);
        updateKD(64);
        updateKD(128);
        
        builder.addSection(".rodata", SHT_PROGBITS, SHF_ALLOC, disp_rodata->addr, rodata_data, 64);
    }
    
    // .text (merged)
    builder.addSection(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 
                       disp_text->addr, disp_text_data, 256);
    
    // .data (function tables)
    builder.addSection(".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                       disp_bss->addr, data_section, 16);
    
    // Write output
    auto output = builder.build();
    
    FILE* out = fopen(output_path.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "Error: Cannot write output: %s\n", output_path.c_str());
        return 1;
    }
    fwrite(output.data(), 1, output.size(), out);
    fclose(out);
    
    printf("Wrote %s: %zu bytes\n", output_path.c_str(), output.size());
    
    return 0;
}
