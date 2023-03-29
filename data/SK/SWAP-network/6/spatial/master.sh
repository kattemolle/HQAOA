for path in */
do
    cd $path
    cp ../110101111100011/run.sh .
    qsub run.sh
    cd ../
done
