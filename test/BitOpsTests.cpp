#include "bitops.h"
#include "gtest/gtest.h"

class BitOpsTestsFixture : public ::testing::Test {
public:
    BitOpsTestsFixture() = default;
    ~BitOpsTestsFixture() override = default;
protected:
    int m_size{0};
    uint32_t m_u32Val{2};
    uint8_t m_u8Val{2};
    std::vector<uint32_t> m_u32Vec{2, 4, 8, 16, 32, 64, 128, 256};
    std::vector<float> m_floatVec{2.0f};
};


TEST_F(BitOpsTestsFixture, DIVUP_Tests) {
    EXPECT_EQ(DIVUP(0, 8), 0);
    EXPECT_EQ(DIVUP(1, 8), 1);
    EXPECT_EQ(DIVUP(7, 8), 1);
    EXPECT_EQ(DIVUP(8, 8), 1);
    EXPECT_EQ(DIVUP(9, 8), 2);
}

TEST_F(BitOpsTestsFixture, ROUNDUP_Tests) {
    EXPECT_EQ(ROUNDUP(0, 8), 0);
    EXPECT_EQ(ROUNDUP(1, 8), 8);
    EXPECT_EQ(ROUNDUP(7, 8), 8);
    EXPECT_EQ(ROUNDUP(8, 8), 8);
    EXPECT_EQ(ROUNDUP(9, 8), 16);
    EXPECT_EQ(ROUNDUP(15, 8), 16);
    EXPECT_EQ(ROUNDUP(16, 8), 16);
    EXPECT_EQ(ROUNDUP(17, 8), 24);
}

TEST_F(BitOpsTestsFixture, ALIGN_POWER_Tests) {
    EXPECT_EQ(ALIGN_POWER(7, 8), 8);
    EXPECT_EQ(ALIGN_POWER(8, 8), 8);
    EXPECT_EQ(ALIGN_POWER(9, 8), 16);
}

TEST_F(BitOpsTestsFixture, ALIGN_SIZE_Tests) {
    m_size = 1;
    ALIGN_SIZE(m_size, 8);
    EXPECT_EQ(m_size, 8);
    m_size = 7;
    ALIGN_SIZE(m_size, 8);
    EXPECT_EQ(m_size, 8);
}

TEST_F(BitOpsTestsFixture, divUp_Tests) {
    EXPECT_EQ(divUp(0, 8), 0);
    EXPECT_EQ(divUp(1, 8), 1);
    EXPECT_EQ(divUp(7, 8), 1);
    EXPECT_EQ(divUp(8, 8), 1);
    EXPECT_EQ(divUp(9, 8), 2);
    EXPECT_EQ(divUp(15, 8), 2);
    EXPECT_EQ(divUp(16, 8), 2);
    EXPECT_EQ(divUp(17, 8), 3);
}

TEST_F(BitOpsTestsFixture, roundUp_Tests) {
    EXPECT_EQ(roundUp(0, 8), 0);
    EXPECT_EQ(roundUp(1, 8), 8);
    EXPECT_EQ(roundUp(7, 8), 8);
    EXPECT_EQ(roundUp(8, 8), 8);
    EXPECT_EQ(roundUp(9, 8), 16);
    EXPECT_EQ(roundUp(15, 8), 16);
    EXPECT_EQ(roundUp(16, 8), 16);
    EXPECT_EQ(roundUp(17, 8), 24);
}

TEST_F(BitOpsTestsFixture, roundDown_Tests) {
    EXPECT_EQ(roundDown(0, 8), 0);
    EXPECT_EQ(roundDown(1, 8), 0);
    EXPECT_EQ(roundDown(7, 8), 0);
    EXPECT_EQ(roundDown(8, 8), 8);
    EXPECT_EQ(roundDown(9, 8), 8);
    EXPECT_EQ(roundDown(15, 8), 8);
    EXPECT_EQ(roundDown(16, 8), 16);
    EXPECT_EQ(roundDown(17, 8), 16);
}

TEST_F(BitOpsTestsFixture, alignUp_Tests) {
    EXPECT_EQ(alignUp(0, 8), 0);
    EXPECT_EQ(alignUp(1, 8), 8);
    EXPECT_EQ(alignUp(7, 8), 8);
    EXPECT_EQ(alignUp(8, 8), 8);
    EXPECT_EQ(alignUp(9, 8), 16);
    EXPECT_EQ(alignUp(15, 8), 16);
    EXPECT_EQ(alignUp(16, 8), 16);
    EXPECT_EQ(alignUp(17, 8), 24);
}


TEST_F(BitOpsTestsFixture, u32fp8MaxValue) {
    EXPECT_EQ(u32fp8MaxValue(), 0xf0000000);
}

TEST_F(BitOpsTestsFixture, u32fp8Encode) {
    uint8_t expected = 2;
    EXPECT_EQ(u32fp8Encode(this->m_u32Val), expected);
}

TEST_F(BitOpsTestsFixture, u32fp8Decode) {
    uint32_t expected = 2;
    EXPECT_EQ(u32fp8Decode(this->m_u8Val), expected);
}

TEST_F(BitOpsTestsFixture, getHash) {
    uint64_t expected = 0xa4495d05731e3337;
    auto ret = getHash(this->m_u32Vec.data(), this->m_u32Vec.size() * sizeof(uint32_t));
    EXPECT_EQ(ret, expected);

    expected = 0xedbaa57e84d6dbaa;
    ret = getHash(this->m_floatVec.data());
    EXPECT_EQ(ret, expected);
}

template <typename T>
class BitOpsTemplateAllIntTestsFixture : public testing::Test {
public:
    BitOpsTemplateAllIntTestsFixture() = default;
    ~BitOpsTemplateAllIntTestsFixture() override = default;
protected:
    T m_countOneBitVal{3};
    T m_firstOneBitVal{3};
    T m_popFirstOneBitVal{3};
    T m_log2DownVal{0};
    T m_log2UpVal{0};
    T m_power2UpVal{0};
    T m_power2DownVal{1};
};

using BitOpsAllIntTypes = ::testing::Types<int, unsigned int, long, unsigned long, long long, unsigned long long>;

TYPED_TEST_SUITE(BitOpsTemplateAllIntTestsFixture, BitOpsAllIntTypes);

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsCountOneBits) {
    EXPECT_EQ(countOneBits(this->m_countOneBitVal), 2);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsFirstOneBits) {
    EXPECT_EQ(firstOneBit(this->m_firstOneBitVal), 0);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, BitOpsPopFirstOneBits) {
    EXPECT_EQ(popFirstOneBit(&(this->m_popFirstOneBitVal)), 0);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, log2Down) {
    EXPECT_EQ(log2Down(this->m_log2DownVal), -1);
    this->m_log2DownVal = 1;
    EXPECT_EQ(log2Down(this->m_log2DownVal), 0);
    this->m_log2DownVal = 2;
    EXPECT_EQ(log2Down(this->m_log2DownVal), 1);
}


TYPED_TEST(BitOpsTemplateAllIntTestsFixture, log2Up) {
    EXPECT_EQ(log2Up(this->m_log2UpVal), 0);
    this->m_log2UpVal = 1;
    EXPECT_EQ(log2Up(this->m_log2UpVal), 0);
    this->m_log2UpVal = 2;
    EXPECT_EQ(log2Up(this->m_log2UpVal), 1);
    this->m_log2UpVal = 3;
    EXPECT_EQ(log2Up(this->m_log2UpVal), 2);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, pow2Up){
    EXPECT_EQ(pow2Up(this->m_power2UpVal), 1);
    this->m_power2UpVal = 1;
    EXPECT_EQ(pow2Up(this->m_power2UpVal), 1);
    this->m_power2UpVal = 2;
    EXPECT_EQ(pow2Up(this->m_power2UpVal), 2);
    this->m_power2UpVal = 3;
    EXPECT_EQ(pow2Up(this->m_power2UpVal), 4);
}

TYPED_TEST(BitOpsTemplateAllIntTestsFixture, pow2Down){
    EXPECT_EQ(pow2Down(this->m_power2DownVal), 1);
    this->m_power2DownVal = 2;
    EXPECT_EQ(pow2Down(this->m_power2DownVal), 2);
    this->m_power2DownVal = 3;
    EXPECT_EQ(pow2Down(this->m_power2DownVal), 2);
}


template <typename T>
class BitOpsTemplateUnsignedTestsFixture : public testing::Test {
public:
    BitOpsTemplateUnsignedTestsFixture() = default;
    ~BitOpsTemplateUnsignedTestsFixture() override = default;
protected:
    T m_reverseSubBits{2};
    T m_reverseBits{2}; // 0b10
};

using BitOpsUnsignedTypes = ::testing::Types<unsigned int, unsigned long, unsigned long long>;

TYPED_TEST_SUITE(BitOpsTemplateUnsignedTestsFixture, BitOpsUnsignedTypes);

TYPED_TEST(BitOpsTemplateUnsignedTestsFixture, reverseSubBits) {
    auto ret = reverseSubBits<TypeParam, 1>(this->m_reverseSubBits);
    EXPECT_EQ(ret, this->m_reverseSubBits);
    ret = reverseSubBits<TypeParam, 2>(this->m_reverseSubBits);
    EXPECT_EQ(ret, 1);
    ret = reverseSubBits<TypeParam, 4>(this->m_reverseSubBits);
    EXPECT_EQ(ret, 4); 
    ret = reverseSubBits<TypeParam, 8>(this->m_reverseSubBits);
    EXPECT_EQ(ret, 64); 
    ret = reverseSubBits<TypeParam, 16>(this->m_reverseSubBits);
    EXPECT_EQ(ret, 16384);
    ret = reverseSubBits<TypeParam, 32>(this->m_reverseSubBits);
    EXPECT_EQ(ret, 1073741824);
}


TYPED_TEST(BitOpsTemplateUnsignedTestsFixture, reverseBits) {
    auto ret = reverseBits(this->m_reverseBits, 2);
    EXPECT_EQ(ret, 1); // 0b01
    ret = reverseBits(this->m_reverseBits, 16);
    EXPECT_EQ(ret, 16384); // 0b0100000000000000
}

