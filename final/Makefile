PROJECT=final
SRC_DIR=src
SRCS=${SRC_DIR}/${PROJECT}.py
REPORT_DIR=report
REPORT=${REPORT_DIR}/report.pdf
DIST_CONTENT=${SRCS} DIST_REBUILD_CONTENT=${REPORT}

PYTHON=/opt/local/bin/python -Wi
DIST_DIR=dist
ZIP=zip -q
LATEX2PDF=/usr/local/texlive/2011/bin/x86_64-darwin/pdflatex -shell-escape
OCTAVE=/opt/local/bin/octave --silent \
                             --eval "addpath(strcat(pwd,'/src'));" \
                             --eval "warning('off', 'print:missing_fig2dev');" \
                             --eval "warning('off', 'print:missing_pstoedit');"
IMAGES=$(notdir $(wildcard images/*.tif))

LC_CTYPE="C"


all: histograms

histograms: $(IMAGES:.tif=_histogram.eps)

%_histogram.mat: images/%.tif results/histograms
	@${PYTHON} src/create_histogram.py $< $(word 2,$^)/$@

%_histogram.eps: %_histogram.mat results/histograms
	@${OCTAVE} --eval "plot_histogram( '$(word 2,$^)/$<', '$(word 2,$^)/$@');" > /dev/null

results/histograms:
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
	@${RM} verslag/*.{aux,log,synctex.gz}

mrproper: clean
	@${RM} verslag/verslag.pdf
