CC      = cc
CFLAGS  = -std=gnu99 -O3 -Wall -g
LDFLAGS = -lm

EXES =
OBJS = mmio.o

ARCH = $(shell uname -p)

all:
	make -C matrices

cg-coo.o: cg.c COO/common.h
	$(CC) $(CFLAGS) -DCOO=1 -c $< -o $@
cg-csr.o: cg.c CSR/common.h
	$(CC) $(CFLAGS) -DCSR=1 -c $< -o $@


define COO_EXE
$(1): $(2) cg-coo.o mmio.o COO/common.o
	$(CC) $$^ -o $$@ $(LDFLAGS)
$(2): COO/common.h COO/ecc.h
EXES += $(1)
OBJS += $(2)
endef

$(eval $(call COO_EXE, cg-coo-c-baseline, COO/c/spmv-baseline.o))
$(eval $(call COO_EXE, cg-coo-c-constraints, COO/c/spmv-constraints.o))
$(eval $(call COO_EXE, cg-coo-c-sed, COO/c/spmv-sed.o))
$(eval $(call COO_EXE, cg-coo-c-sec7, COO/c/spmv-sec7.o))
$(eval $(call COO_EXE, cg-coo-c-sec8, COO/c/spmv-sec8.o))
$(eval $(call COO_EXE, cg-coo-c-secded, COO/c/spmv-secded.o))

ifneq (,$(findstring armv7,$(ARCH)))
  $(eval $(call COO_EXE, cg-coo-arm32-sed, COO/arm32/spmv-sed.o))
endif


define CSR_EXE
$(1): $(2) cg-csr.o mmio.o CSR/common.o
	$(CC) $$^ -o $$@ $(LDFLAGS)
$(2): CSR/common.h CSR/ecc.h
EXES += $(1)
OBJS += $(2)
endef

$(eval $(call CSR_EXE, cg-csr-c-baseline, CSR/c/spmv-baseline.o))
$(eval $(call CSR_EXE, cg-csr-c-sed, CSR/c/spmv-sed.o))
$(eval $(call CSR_EXE, cg-csr-c-sec7, CSR/c/spmv-sec7.o))
$(eval $(call CSR_EXE, cg-csr-c-sec8, CSR/c/spmv-sec8.o))
$(eval $(call CSR_EXE, cg-csr-c-secded, CSR/c/spmv-secded.o))

ifneq (,$(findstring armv7,$(ARCH)))
  $(eval $(call CSR_EXE, cg-csr-arm32-sed, CSR/arm32/spmv-sed.o))
endif


all: $(EXES)

test: $(EXES)
	for exe in $(EXES); do \
	  ./$$exe -b 5 >/dev/null ; \
		if [ $$? -ne 0 ]; then echo "FAILED $$exe"; else echo "passed $$exe"; fi ; \
	done \

clean:
	rm -f $(EXES) $(OBJS) cg-coo.o cg-csr.o COO/common.o CSR/common.o

.PHONY: clean test
