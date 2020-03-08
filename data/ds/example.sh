
# Input file and output intermediate files
INFILE=el_1k.txt
GOALSFILE=el_1k.goals.txt
GOODFILE=el_1k.good.txt
ANNOTATEDFILE=el_1k.annotated.txt
NEWTEMPLATESFILE=el_1k.templates.tsv

#python3 strip_entity_annotations.py $INFILE $GOALSFILE $GOODFILE

# el_1k.good has the sentences to be annotated
UDPIPE_ROOT=/Users/antonis/research/mbert_analysis/
UDPIPE_MODEL=$UDPIPE_ROOT/udpipe_models/greek-gdt-ud-2.4-190531.udpipe

# Annotate the sentences
#python3 $UDPIPE_ROOT/run_udpipe.py horizontal conllu $UDPIPE_MODEL< $GOODFILE > $ANNOTATEDFILE

# Convert the annotated sentences into templates
python3 convert_mined_to_templates.py $ANNOTATEDFILE $GOALSFILE $NEWTEMPLATESFILE
