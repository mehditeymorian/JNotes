#!/bin/bash

# <row Id="1320292" PostId="429210" VoteTypeId="5" UserId="3897" CreationDate="2009-01-10T00:00:00.000" />


# output: id postId voteTypeId userId

cat votes.xml | grep "UserId" | perl -ne 'print "$1 $2 $3 $4\n" if /Id="(\d*)"\sPostId="(\d*)"\sVoteTypeId="(\d*)"\sUserId="(\d*)"/ ' >> result.txt

