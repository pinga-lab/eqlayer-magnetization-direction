help:
	@echo "Available makes:"
	@echo "manuscript - compile the latex files to build the manuscript"
	@echo "clean - remove unused files"

all:
	make manuscript
	make clean

# Compile the manuscript
manuscript:
	pdflatex eqlayer_magdir_manuscript.tex
	bibtex references
	pdflatex eqlayer_magdir_manuscript.tex
	pdflatex eqlayer_magdir_manuscript.tex

# Remove unused files
clean:
	rm *.aux *.bbl *.blg *.log *.pdf *.lof *.lot *.synctex.gz *.out

