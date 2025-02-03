help:
	@echo ''
	@echo 'make intel    Build with ifort. (Intel compilers)'
	@echo 'make nag      Build with nagfor. (NAG compilers)'
	@echo 'make gfortran Build with gfortran. (GNU compilers)'
	@echo 'make nvhpc    Build with nvhpc. (Nvidia compilers)'
	@echo 'make aurora   Build with nfort. (NEC Aurora compilers)'
	@echo ''

.DEFAULT:
	@test -f Makefile.$@
	$(MAKE) -f Makefile.$@

clean:
	rm -f *.o pi

.PHONY: clean, help

