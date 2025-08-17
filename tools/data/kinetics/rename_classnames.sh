#!/usr/bin/env bash

# Rename classname for convenience
DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../../../data/${DATASET}/
ls ./train_original | while read class; do \
  newclass=`echo $class | tr ")" "-" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "train_original/${class}" "train_original/${newclass}";
  fi
done

ls ./val_original | while read class; do \
  newclass=`echo $class | tr ")" "-" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "val_original/${class}" "val_original/${newclass}";
  fi
done

cd ../../tools/data/kinetics/
