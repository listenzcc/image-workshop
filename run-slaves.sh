#!/bin/sh

pwd=`pwd`

cd $pwd/slaves/BoostingMonocularDepth; ls
cd $pwd/slaves/informative-drawings; ls 

cd $pwd/slaves/BoostingMonocularDepth; python run-server.py &
cd $pwd/slaves/informative-drawings; python run-server.py & 
