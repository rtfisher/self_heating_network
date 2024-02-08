# Makefile for compiling Helmholtz equation solver

# Compiler settings
FC = gfortran
FCFLAGS = 
OUTFILE = helmholtz.exe

# Source files
SOURCES = helmholtz_wrapper.f90 helmholtz_library.F90 main.F90 invert_helm_pt.f90

# Object files
#OBJECTS = $(SOURCES:.f90=.o) $(SOURCES:.F90=.o)
OBJECTS = helmholtz_wrapper.o helmholtz_library.o main.o invert_helm_pt.o

#
# Dependency files
DEK_FILES = const.dek helm_table_storage.dek implno.dek vector_eos.dek

# Rule to make everything
all: $(OUTFILE)

# Rule to link the program
$(OUTFILE): $(OBJECTS)
	$(FC) -o $@ $^

# Rule to compile source files
%.o: %.f90 $(DEK_FILES)
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.F90 $(DEK_FILES)
	$(FC) $(FCFLAGS) -c $< -o $@

# Clean the build directory
clean:
	rm -f $(OBJECTS) $(OUTFILE)

# Phony targets
.PHONY: all clean

