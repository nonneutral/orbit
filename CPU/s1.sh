#!/bin/bash
./outfiles/a.out
paste ./outfiles/xplot.txt ./outfiles/yplot.txt > ./outfiles/xy
paste ./outfiles/tplot.txt ./outfiles/rplot.txt > ./outfiles/tr
paste ./outfiles/tplot.txt ./outfiles/kinetic.txt ./outfiles/potential.txt > ./outfiles/tUU
gnuplot './outfiles/script01'
gnuplot './outfiles/script02'
gnuplot './outfiles/script03'

