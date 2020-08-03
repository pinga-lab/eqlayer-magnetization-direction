help:
	@echo "Available makes:"
	@echo "figs - convert figures to TIF/CMYK"
	@echo "clean - remove unused files"

all:
	make figs
	make clean

# Convert and rename figures
figs:
	convert Fig/Fig1.png -colorspace CMYK -units PixelsPerInch Fig/Figure1.tiff
	convert Fig/unidir_test/Fig2.png -colorspace CMYK -units PixelsPerInch Fig/Figure2.tiff
	convert Fig/unidir_shallow_test/Fig3.png -colorspace CMYK -units PixelsPerInch Fig/Figure3.tiff
	convert Fig/unidir_shallow_diff_test/Fig4.png -colorspace CMYK -units PixelsPerInch Fig/Figure4.tiff
	convert Fig/field_data_montes_claros/Fig5.png -colorspace CMYK -units PixelsPerInch Fig/Figure5.tiff
	convert Fig/field_data_montes_claros/Fig6.png -colorspace CMYK -units PixelsPerInch Fig/Figure6.tiff
	convert Fig/Fig7.png -colorspace CMYK -units PixelsPerInch Fig/Figure7.tiff

# Remove unused files
clean:
	rm *.aux *.bbl *.blg *.log *.pdf *.lof *.lot *.synctex.gz *.out

