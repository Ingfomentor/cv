PROJECT=final
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

SRC_DIR=images
IMAGES=$(notdir $(wildcard ${SRC_DIR}/*.tif))

LC_CTYPE="C"

VPATH = ${SRC_DIR}

all: clean enhanced_histograms histograms

histograms: $(IMAGES:.tif=_histogram.eps)
enhanced_images: $(IMAGES:.tif=_enhanced.tif)
enhanced_histograms: $(IMAGES:.tif=_enhanced_histogram.eps)

%_histogram.mat: %.tif
	@echo "--- creating histogram for $<"
	@${PYTHON} src/create_histogram.py $< $@

%_histogram.eps: %_histogram.mat
	@echo "--- creating histogram image for $<"
	@${OCTAVE} --eval "plot_histogram( '$<', '$@');" > /dev/null

%_enhanced.tif: %.tif %_histogram.mat
	@echo "--- creating enhanced version of $<"
	@${PYTHON} src/create_enhanced_image.py $< $(word 2,$^) $@

results/histograms:
	@mkdir -p $@

results/${SRC_DIR}:
	@mkdir -p $@

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
	@${RM} *.{mat,eps,tif}
	@${RM} verslag/*.{aux,log,synctex.gz}

mrproper: clean
	@${RM} verslag/verslag.pdf

.PRECIOUS: %_histogram.mat %_histogram.eps %_enhanced.tif
.SECONDARY: %_histogram.mat %_histogram.eps %_enhanced.tif
