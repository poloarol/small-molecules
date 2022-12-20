
# cd into data directory and execute shell script

$source_file = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
$destination_file = "data/qm9.csv"

wget -c -P $destination_file $source_file
