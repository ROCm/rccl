#!/usr/bin/gawk -f
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.# Usage:

BEGIN {
  max_rank=0
  rings[""]=0
  max_ring=0
  treedns[""]=0
  max_treedn=0
  conn[""]=0
  col_start=2
  col_p1=col_start+1
  col_p3=col_start+3
  col_p4=col_start+4
  col_p5=col_start+5
  col_p6=col_start+6
  col_p7=col_start+7
  col_p8=col_start+8
}

{
  if($3=="NCCL" && $4=="INFO" && col_start==2) {
    col_start=5
    col_p1=col_start+1
    col_p3=col_start+3
    col_p4=col_start+4
    col_p5=col_start+5
    col_p6=col_start+6
    col_p7=col_start+7
    col_p8=col_start+8
  }

  if($col_start=="Ring" && $col_p4="->" && $col_p6=="->") {
    chan=strtonum($col_p1)
    rank=strtonum($col_p5)
    next_rank=strtonum($col_p7)
    rings[rank "," next_rank "," chan]="1"
    if(chan>max_ring)
      max_ring=chan
    if(rank>max_rank)
      max_rank=rank
  }

  if($col_start=="Trees") {
    col_1=col_start+1
    col_2=col_start+2
    do {
      match($col_1, /\[([0-9]+)\]/, ary)
      chan=strtonum(ary[1])
      match($col_2, /(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\|(\-?[0-9]+)\->(\-?[0-9]+)\->(\-?[0-9]+)\/(\-?[0-9]+)\/(\-?[0-9]+)/, ary)
      if(ary[8]!="-1")
        treedns[ary[7] "," ary[8] "," chan]="1"
      if(ary[9]!="-1")
        treedns[ary[7] "," ary[9] "," chan]="1"
      if(ary[10]!="-1")
        treedns[ary[7] "," ary[10] "," chan]="1"
      if(chan>max_treedn)
        max_treedn=chan
      col_1=col_1+2
      col_2=col_2+2
    } while ($col_1!="")
  }

  if($col_p6=="via") {
    match($col_p1, /([0-9]+)/, ary)
    chan=strtonum(ary[1])
    match($col_p3, /([0-9]+)\[.*\]/, ary)
    s=ary[1]
    match($col_p5, /([0-9]+)\[.*\]/, ary)
    d=ary[1]
    conn[s "," d "," chan]=$col_p7
  }

  if($col_p6=="[receive]" && $col_p7=="via") {
    match($col_p1, /([0-9]+)/, ary)
    chan=strtonum(ary[1])
    match($col_p3, /([0-9]+)\[.*\]/, ary)
    s=ary[1]
    match($col_p5, /([0-9]+)\[.*\]/, ary)
    d=ary[1]
    conn[s "," d "," chan]=$col_p8
  }
}

END {
  printf "digraph RCCL {\n"
  for(r=0;r<max_treedn+1;r++) {
    printf "  subgraph tree_%d {\n", r
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in treedns) {
          val=conn[s "," d "," r]
          style="solid"
          color="red"
          if (match(val,"NET")) {
            style="dashed"
            if (match(val,"GDRDMA"))
              color="green"
          }
          if (match(val,"P2P")) {
            color="green"
          }
          printf "    t%d_%d -> t%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, s, r, d, val, color, style
        }
      }
    }
    printf "\n"
    for(s=0;s<=max_rank;s++) {
      printf "    t%d_%d [label=\"%d\",fontsize=\"28\"];\n", r, s, s
    }
    printf "  }\n\n"
  }

  for(r=0;r<max_ring+1;r++) {
    remove_ring=0
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in rings && !((s "," d "," r) in conn)) {
          remove_ring=1
          break;
        }
      }
      if(d<=max_rank)
        break;
    }
    if (remove_ring!=0)
      continue
    printf "  subgraph ring_%d {\n", r
    for(s=0;s<=max_rank;s++) {
      for(d=0;d<=max_rank;d++) {
        if ((s "," d "," r) in rings) {
          val=conn[s "," d "," r]
          style="solid"
          color="red"
          if (match(val,"NET")) {
            style="dashed"
            if (match(val,"GDRDMA"))
              color="green"
          }
          if (match(val,"P2P")) {
            color="green"
          }
          printf "    r%d_%d -> r%d_%d [label=\"%s\",color=\"%s\",style=\"%s\",fontname=\"Helvetica\"];\n", r, s, r, d, val, color, style
        }
      }
    }
    printf "\n"
    for(s=0;s<=max_rank;s++) {
      printf "    r%d_%d [label=\"%d\",fontsize=\"28\"];\n", r, s, s
    }
    printf "  }\n\n"
  }
  printf "}\n"
}
