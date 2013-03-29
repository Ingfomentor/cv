PROJECT=face_recognition
SRCS=${PROJECT}.py

PYTHON=/opt/local/bin/python
DIST_DIR=dist
ZIP=zip -q

all: run-${PROJECT}

dist:
	@mkdir ${DIST_DIR}
	@cp ${SRCS} ${DIST_DIR}
	@(cd ${DIST_DIR}/; ${ZIP} -r ../christophe_van_ginneken.zip *)
	@rm -rf ${DIST_DIR}

run-%: %.py
	@${PYTHON} $<
