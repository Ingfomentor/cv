PROJECT=final

# configuration
CROP_WIDTH=700						# center width to retain when cropping
CROP_HEIGHT=1100					# center height to retain when cropping
CROP_TOP_OFFSET=250				# additional top offset for cropping
ALPHA=20									# alpha value for contrast-stretching
SLICES=5									# 5 slices = 4 teeth
EXPECTED_JAW_SPLIT=0.7		# the split is roughly 20% below the horizontal center
SIGMA=0.4									# "peakyness" of the Gaussian prob. dist
INVERSION_TOP=1.1					# top of prob. dist.
UPPER_LENGTH=350					# length of upper-jaw teeth
LOWER_LENGTH=300 					# length of lower-jaw teeth

# !!! DON'T change anything below this point (unless you know what you're doing)

SRC_DIR=src
SRCS=${SRC_DIR}/*.{py,m}
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
MONTAGE=montage -label "%f" -geometry "140x170>+4+3" -tile 5x
MONTAGE_GRAPH=montage -label "%f" -geometry "200x150>+4+3" -tile 5x
CONVERT=convert

SRC_DIR=images
IMAGES=$(notdir $(wildcard ${SRC_DIR}/*.tif))
SEGMENTS=$(notdir $(wildcard ${SRC_DIR}/segmented/*.png))

LC_CTYPE="C"

VPATH = ${SRC_DIR}


all: overviews

overviews: cropped.png cropped_hists.png sigmoids.png \
					 enhanced.png enhanced_hists.png \
					 jaw_splits.png jaw_splits_hists.png \
					 teeth_iso.png teeth_iso_hists.png roi.png \
					 contours.png masks.png \
					 comparison-0.png comparison-1.png \
					 crown_histograms-0.png crown_histograms-1.png

cropped.png: $(IMAGES:.tif=_crp.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
cropped_hists.png: $(IMAGES:.tif=_crp_hist.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} $^ $@
enhanced.png: $(IMAGES:.tif=_crp_enh.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
enhanced_hists.png: $(IMAGES:.tif=_crp_enh_hist.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} $^ $@
sigmoids.png: $(IMAGES:.tif=_crp_enh_sigmoid.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} $^ $@
jaw_splits.png: $(IMAGES:.tif=_crp_enh_jaw_split.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
jaw_splits_hists.png: $(IMAGES:.tif=_crp_enh_jaw_split_hists.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} $^ $@
teeth_iso.png: $(IMAGES:.tif=_crp_enh_teeth_iso.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
teeth_iso_hists.png: $(IMAGES:.tif=_crp_enh_teeth_iso_hists.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} $^ $@
roi.png: $(IMAGES:.tif=_crp_enh_roi.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
contours.png: $(IMAGES:.tif=_crp_enh_contours.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
masks.png: $(IMAGES:.tif=_crp_enh_masks.tif)
	@echo "--- creating montage $@"
	@${MONTAGE} $^ $@
comparison.png: ${SEGMENTS}
	@echo "--- creating montage $@"
	@${MONTAGE} -tile 8x $^ $@
comparison-0.png comparison-1.png: comparison.png
	@echo "--- splitting $<"
	@${CONVERT} -crop 1184x1940 $< $<
crown_histograms.png: $(SEGMENTS:.png=_crown_hist.eps)
	@echo "--- creating montage $@"
	@${MONTAGE_GRAPH} -tile 8x $^ $@
crown_histograms-0.png crown_histograms-1.png: crown_histograms.png
	@echo "--- splitting $<"
	@${CONVERT} -crop 1664x1740 $< $<


%_crp.tif: %.tif
	@echo "--- cropping image $<"
	@${PYTHON} src/crop_image.py $< $@ ${CROP_WIDTH} ${CROP_HEIGHT} ${CROP_TOP_OFFSET}

%_hist.mat: %.tif
	@echo "--- creating hist for $<"
	@${PYTHON} src/create_histogram.py $< $@

%_hist.eps: %_hist.mat
	@echo "--- visualizing hist $<"
	@${OCTAVE} --eval "plot_histogram( '$<', '$@');" > /dev/null

%_enh.mat: %_hist.mat
	@echo "--- determining enhancement parameters of $<"
	@${PYTHON} src/determine_enhancement_parameters.py $^ ${ALPHA} $@

%_enh.tif: %.tif %_enh.mat
	@echo "--- creating enhanced version of $<"
	@${PYTHON} src/create_enhanced_image.py $^ $@

%_sigmoid.eps: %.mat
	@echo "--- visualising contrast-stretching transformation from $<"
	@${OCTAVE} --eval "plot_sigmoid( '$<', '$@');" > /dev/null

%_jaw_split.mat: %.tif
	@echo "--- detecting jaw split for $<"
	@${PYTHON} src/jaw_split.py $< ${SLICES} ${EXPECTED_JAW_SPLIT} ${SIGMA} ${INVERSION_TOP} $@ 

%_jaw_split_hists.eps: %_jaw_split.mat
	@echo "--- visualizing jaw split hists from $<"
	@${OCTAVE} --eval "plot_jaw_histograms( '$<', '$@');" > /dev/null

%_jaw_split.tif: %.tif %_jaw_split.mat
	@echo "--- visualizing jaw split for $<"
	@${PYTHON} src/visualize_jaw_split.py $^ $@

%_teeth_iso.mat: %.tif %_jaw_split.mat
	@echo "--- isolating incisors for $<"
	@${PYTHON} src/teeth_isolation.py $^ ${UPPER_LENGTH} ${LOWER_LENGTH} $@
	
%_teeth_iso_hists.eps: %_teeth_iso.mat
	@echo "--- visualizing teeth isolation hists from $<"
	@${OCTAVE} --eval "plot_teeth_isolation_histograms( '$<', '$@');" > /dev/null

%_teeth_iso.tif: %.tif %_jaw_split.mat %_teeth_iso.mat
	@echo "--- visualizing incisors for $<"
	@${PYTHON} src/visualize_teeth_isolation.py $^ $@

%_roi.mat: %_teeth_iso.mat
	@echo "--- creating ROI from $<"
	@${PYTHON} src/roi.py $< $@

%_roi.tif: %.tif %_teeth_iso.mat %_roi.mat
	@echo "--- visualizing ROI for $<"
	@${PYTHON} src/visualize_roi.py $^ $@

%_contours.mat: %.tif %_roi.mat
	@echo "--- creating contours for $<"
	@${PYTHON} src/contours.py $^ $@

%_contours.tif: %.tif %_contours.mat %_roi.mat
	@echo "--- visualizing contours on $<"
	@${PYTHON} src/visualize_contours.py $^ $@

%_masks.mat: %.tif %_contours.mat %_roi.mat
	@echo "--- creating masks for $<"
	@${PYTHON} src/mask.py $^ $@

%_masks.tif: %.tif %_masks.mat %_roi.mat
	@echo "--- visualizing masks on $<"
	@${PYTHON} src/visualize_masks.py $^ $@

# UGLY AS HELL - but it allows for prerequisites ;-)

define SEGMENT
%-$(1).png: ${SRC_DIR}/segmented/%-$(1).png %_crp_enh_masks.mat %_crp_enh_roi.mat
	@echo "--- creating comparing mask for $$@"
	@${PYTHON} src/compare_mask.py $$^ $$@ $(1) ${CROP_WIDTH} ${CROP_HEIGHT} ${CROP_TOP_OFFSET}
endef

$(foreach i,0 1 2 3 4 5 6 7,$(eval $(call SEGMENT,$(i))))

define CROWN
%-$(1)_crown_hist.eps: %_crp_enh_contours.mat
	@echo "--- creating crown histogram $(1) from $$<"
	@${OCTAVE} --eval "plot_crown_histogram('$$<', $(1), '$$@')" > /dev/null
endef

$(foreach i,0 1 2 3 4 5 6 7,$(eval $(call CROWN,$(i))))


ui:
	@${PYTHON} src/ui.py ${CROP_WIDTH} ${CROP_HEIGHT} ${CROP_TOP_OFFSET} ${ALPHA} \
											 ${EXPECTED_JAW_SPLIT} ${SIGMA} ${INVERSION_TOP} \
											 ${UPPER_LENGTH} ${LOWER_LENGTH}

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
	@${RM} *.{png,eps,tif,pdf}
	@${RM} verslag/*.{aux,log,synctex.gz}

mrproper: clean
	@${RM} *.mat
	@${RM} verslag/verslag.pdf

.PRECIOUS: %_hist.mat %_hist.eps %_enh.mat %_jaw_split.mat %_teeth_iso.mat %_roi.mat %_contours.mat %_masks.mat
.SECONDARY: %_hist.mat %_hist.eps %_enh.mat %_jaw_split.mat %_teeth_iso.mat %_roi.mat %_contours.mat %_masks.mat
