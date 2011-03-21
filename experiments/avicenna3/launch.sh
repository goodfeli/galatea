# Create jobs from the current directory
if [ ${0:0:1} = "/" ]; then
	cd $(dirname $0)
else
	cd $(dirname "$PWD/$0")
fi

TABLE='AVICENNA3_DAE'
python -c "import experiments.avicenna2.script as c; c.create('$TABLE')"

# Launch jobs from another directory
cd /data/lisatmp/ift6266h11/$LOGNAME
URL=postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/$TABLE
jobdispatch --repeat_jobs=16 --mem=2000 jobman sql -n0 $URL ./

# Reset job status
# jobman sqlstatus --all --set_status=START $URL
