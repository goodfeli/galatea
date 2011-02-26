# Go to the current directory
if [ ${0:0:1} = "/" ]; then
	cd $(dirname $0)
else
	cd $(dirname "$PWD/$0")
fi

# Update view
TABLE='AVICENNA3_DAE'
python -c "import experiments.avicenna2.script as c; c.update_view('$TABLE')" > /dev/null

# Retrieving the results
COL="id,
n_hid,
act_enc,
act_dec,
corruption_class,
corruption_level,
base_lr,
epochs,
batch_size,
error_valid,
error_test,
best_alc_value,
best_alc_epoch"
COL=$(echo $COL | sed 's/\_//g')

CMD="select ${COL} from ${TABLE}view order by bestalcvalue;"

psql -U ift6266h11 -d ift6266h11_sandbox_db -h gershwin.iro.umontreal.ca -c "$CMD"
