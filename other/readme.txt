The files contain hourly averages of the magnetic field for the in-ecliptic 
cruise, timed at mid-interval (i.e. on the half hour). There are eight columns:
	1) Year (two digit integer)
	2) Decimal day no. (January 1 = Day 1, fractional part completely 
	   specifies the timing of the data point)
	3) Decimal hour (actually redundant but aids readability of the file)
	4) Magnetic field hour average of R component (nT)
	5) Magnetic field hour average of T component (nT)
	6) Magnetic field hour average of N component (nT)
	7) Magnetic field hour average magnitude (nT) (note that this is the 
           average of the magnitudes of the individual full resolution vectors 
           as opposed to the magnitude of the vector formed by the hourly
           averaged components) 
	8) Number of full resolution vectors that have contributed to the 
           average (as a crude statistical validity check)

Data gaps are not flagged in any way - if there were no available data in a 
particular hour then there is no entry for that hour in the file.
