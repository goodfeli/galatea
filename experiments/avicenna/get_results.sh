if [ -e "./results" ];then
    mv ./results ./results.back
fi
python -c "import experiments.create_db as c; c.update_view('AVI5_')"
psql -U ift6266h11 -d ift6266h11_sandbox_db -h gershwin.iro.umontreal.ca -f psql_queries.sql > ./results
echo "results dumped in 'results'"

