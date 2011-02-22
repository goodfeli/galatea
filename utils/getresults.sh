#FILE!/bin/bash

SETS="avicenna harry rita sylvester terry"
URL="http://www.causality.inf.ethz.ch/unsupervised-learning.php?page=leaderboard&model="

SAVE_DIR=/data/lisa/data/UTLC/results
DATE=`date +%m-%d-%k-%M`
PID_FILE=$SAVE_DIR/getresults.pid

if [ -e $PID_FILE ]; then
    echo "pidfile $PID_FILE exists"
    echo "the script is either running or had a unclean exit"
    exit 1
fi

touch $PID_FILE

for set in $SETS; do

    TMP=`mktemp`
    FILE=$SAVE_DIR/$set

    # peu importe le nombre de vierges sacrifiees, le dieu du software-engineering
    # me reserve une place en enfer.
    curl -s $URL$set | egrep -A7 '<td valign="top">[0-9]*<br>' | head -45 | \
        egrep '<td valign="top">.*</td>' | \
        sed -e '/<td valign="top"><\/td>/d' -e 's/.*<td valign="top">//' \
            -e 's/<\/td>//g' -e '/xxxxxx/d' -e '/^[ \t]*$/d' | tr -s '\r\n' '\n'| \
            awk -F '\n' 'BEGIN{RS=""}{for(i=1;i<=NF;i=i+4){printf("%-15s%-15s%-15s%-15s\n", $i, $(i+1), $(i+2), $(i+3))}}' >> $FILE

    sort -r -n -k 3 -u $FILE > $TMP
    mv $TMP $FILE

    chmod 644 $FILE

done

rm $PID_FILE
