n_threads=`${1} `
cd /tmp

# extract the fasta
selected=$(awk '{print $1}' for_subPICRUSt2.tsv)
for taxa in $selected
do
    grep -A 1 $taxa ASV.fasta >> subASV.fasta
done

mkdir PICRUSt2
mv ./subASV.fasta ./PICRUSt2/subASV.fasta

cd PICRUSt2

echo 'Step 1. Align ASVs to reference sequences'
place_seqs.py -s subASV.fasta -o out_tree -p $n_threads  --intermediate place_seqs

echo 'Step 2. Place ASVs into reference tree'
hsp.py -i 16S -t out_tree -o marker_predicted_and_nsti.tsv.gz -p $n_threads -n
hsp.py -i KO -t out_tree -o KO_predicted.tsv.gz -p $n_threads

mkdir KO
echo 'Step 3. Determine gene family abundance per sample'
metagenome_pipeline.py -i for_subPICRUSt2.tsv -m marker_predicted_and_nsti.tsv.gz -f KO_predicted.tsv.gz  -o KO --strat_out
pathway_pipeline.py -i KO/pred_metagenome_contrib.tsv.gz -o KO -p $n_threads
add_descriptions.py -i KO/pred_metagenome_unstrat.tsv.gz -m KO -o KO/pred_metagenome_unstrat_descrip.tsv.gz