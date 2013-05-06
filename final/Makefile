PROJECT=final

# configuration
CROP_WIDTH=700
CROP_HEIGHT=1100
SLICES=5
EXPECTED_JAW_SPLIT=0.7
RHO=0.4
INVERSION_TOP=1.1

# !!! DON'T change anything below this point (unless you know what you're doing)

SRC_DIR=src
SRCS=${SRC_DIR}/${PROJECT}.py
REPORT_DIR=report
REPORT=${REPORT_DIR}/report.pdf
DIST_CONTENT=${SRCS} DIST_REBUILD_CONTENT=${REPORT}

PYTHON=/opt/local/bin/python -Wi
DIST_DIR=dist
ZIP=zip -q
RM=rm -rf
LATEX2PDF=/usr/local/texlive/2011/bin/x86_64-darwin/pdflatex -shell-escape
OCTAVE=/opt/local/bin/octave --silent \
                             --eval "addpath(strcat(pwd,'/src'));" \
                             --eval "warning('off', 'print:missing_fig2dev');" \
                             --eval "warning('off', 'print:missing_pstoedit');"
MONTAGE=montage -label "%f" -geometry "140x220>+4+3"

SRC_DIR=images
IMAGES=$(notdir $(wildcard ${SRC_DIR}/*.tif))

LC_CTYPE="C"

VPATH = ${SRC_DIR}


all: clean jaw_splits.png teeth_iso.png roi.png

crp_images: $(IMAGES:.tif=_crp.tif)
crp_histograms: $(IMAGES:.tif=_crp_histogram.eps)
crp_enh_images: $(IMAGES:.tif=_crp_enh.tif)
crp_enh_histograms: $(IMAGES:.tif=_crp_enh_histogram.eps)
jaw_splits.png: $(IMAGES:.tif=_crp_enh_jaw_split.tif)
	@echo "--- creating montage for $@"
	@${MONTAGE} $^ $@
teeth_iso.png: $(IMAGES:.tif=_crp_enh_teeth_iso.tif)
	@echo "--- creating montage for $@"
	@${MONTAGE} $^ $@
roi.png: $(IMAGES:.tif=_crp_enh_roi.tif)
	@echo "--- creating montage for $@"
	@${MONTAGE} $^ $@

%_crp.tif: %.tif
	@echo "--- cropping image $<"
	@${PYTHON} src/crop_image.py $< $@ ${CROP_WIDTH} ${CROP_HEIGHT}

%_histogram.mat: %.tif
	@echo "--- creating histogram for $<"
	@${PYTHON} src/create_histogram.py $< $@

%_histogram.eps: %_histogram.mat
	@echo "--- visualizing histogram $<"
	@${OCTAVE} --eval "plot_histogram( '$<', '$@');" > /dev/null

%_jaw_split_histograms.eps: %_jaw_split.mat
	@echo "--- visualizing jaw split histograms $<"
	@${OCTAVE} --eval "plot_jaw_histograms( '$<', '$@');" > /dev/null

%_enh.tif: %.tif %_histogram.mat
	@echo "--- creating enhanced version of $<"
	@${PYTHON} src/create_enhanced_image.py $^ $@

%_jaw_split.mat: %.tif
	@echo "--- detecting jaw split for $<"
	@${PYTHON} src/jaw_split.py $< ${SLICES} ${EXPECTED_JAW_SPLIT} ${RHO} ${INVERSION_TOP} $@ 

%_jaw_split.tif: %.tif %_jaw_split.mat
	@echo "--- visualizing jaw split for $<"
	@${PYTHON} src/visualize_jaw_split.py $^ $@

%_teeth_iso.mat: %.tif %_jaw_split.mat
	@echo "--- isolating incisors for $<"
	@${PYTHON} src/teeth_isolation.py $^ $@

%_teeth_iso.tif: %.tif %_jaw_split.mat %_teeth_iso.mat
	@echo "--- visualizing incisors for $<"
	@${PYTHON} src/visualize_teeth_isolation.py $^ $@

%_roi.mat: %_teeth_iso.mat
	@echo "--- creating ROI from $<"
	@${PYTHON} src/roi.py $< $@

%_roi.tif: %.tif %_jaw_split.mat %_teeth_iso.mat %_roi.mat
	@echo "--- visualizing ROI for $<"
	@${PYTHON} src/visualize_roi.py $^ $@

${REPORT}:
	@(cd report; \
		export PATH=$$PATH:/usr/local/bin ; \
		${LATEX2PDF} report.tex; \
		touch report.tex; \
		${LATEX2PDF} report.tex)

dist: 
	@mkdir ${DIST_DIR}
	@cp ${DIST_CONTENT} ${DIST_DIR}
	@(cd ${DIST_DIR}/; ${ZIP} -r ../christophe_van_ginneken.zip *)
	@rm -rf ${DIST_DIR}

clean:
	@${RM} *.{mat,eps,tif,png}
	@${RM} verslag/*.{aux,log,synctex.gz}

mrproper: clean
	@${RM} verslag/verslag.pdf

.PRECIOUS: %_histogram.mat %_histogram.eps %_enh.tif %_jaw_split.mat %_teeth_iso.mat %_roi.mat
.SECONDARY: %_histogram.mat %_histogram.eps %_enh.tif %_jaw_split.mat %_teeth_iso.mat %_roi.mat
