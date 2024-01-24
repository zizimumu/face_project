#!/bin/bash


while read line;do
	eval "$line"
done < config

echo $MXFC_GPIO_DELAY
echo $MXFC_USB
