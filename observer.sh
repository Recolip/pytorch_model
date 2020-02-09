#!/bin/bash
DIR="../uploader/public/inputs/input"
inotifywait -m -r -e create "$DIR" | while read f

do
    sleep 1
    # you may want to release the monkey after the test 🙂
    python forwardPass.py
    rm ../uploader/public/inputs/input/*
#    echo "Waiting for output folder..."
#    sleep 5
#    echo "Post to url"
#    cd ../output/
#    curl -X POST -H "Content-Type: application/json" -d @../output/output.json https://bitter-bullfrog-2.localtunnel.me/result
#    rm ../output/output.json
#    echo "Done Posting"
#    break 
    # <whatever_command_or_script_you_liketorun>
done
