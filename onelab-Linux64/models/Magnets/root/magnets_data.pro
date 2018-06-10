DefineConstant[
  NumMagnets = {19, Min 1, Max 50, Step 1, Name "Parameters/0Number of magnets"}
];
mm = 1.e-3;

//x = -1;
y = 0;
z = 0;

radius = 0.0201604646760;
height = 20 * mm;

//x = {-0.04112092,  0.04112092, -0.11234446,  0.11234446, -0.04112092,        0.04112092, -0.1534654 ,  0.1534654 , -0.08224186,  0.        ,        0.08224186, -0.1534654 ,  0.1534654 , -0.04112092,  0.04112092,       -0.11234446,  0.11234446, -0.04112092,  0.04112092};
//z = {-1.53465398, -1.53465398, -1.12344468, -1.12344468, -0.71223538,       -0.71223538, -0.4112093 , -0.4112093 ,  0.        ,  0.        ,        0.        ,  0.4112093 ,  0.4112093 ,  0.71223538,  0.71223538,        1.12344468,  1.12344468,  1.53465398,  1.53465398};

//x = {-0.02056046,  0.02056046, -0.05617223,  0.05617223, -0.02056046, 0.02056046, -0.0767327 ,  0.0767327 , -0.04112093,  0.        , 0.04112093, -0.0767327 ,  0.0767327 , -0.02056046,  0.02056046, -0.05617223,  0.05617223, -0.02056046,  0.02056046};
//z = {-0.0767327 , -0.0767327 , -0.05617223, -0.05617223, -0.03561177, -0.03561177, -0.02056046, -0.02056046,  0.        ,  0.       , 0.        ,  0.02056046,  0.02056046,  0.03561177,  0.03561177, 0.05617223,  0.05617223,  0.0767327 ,  0.0767327};


//x = {-0.20560465,   0.20560465,  -0.56172234,   0.56172234, -0.20560465,   0.20560465,  -0.76732699,   0.76732699,  -0.41120929,   0.        ,   0.41120929,  -0.76732699, 0.76732699,  -0.20560465,   0.20560465,  -0.56172234, 0.56172234,  -0.20560465,   0.20560465};
//z = {-0.76732699,  -0.76732699,  -0.56172234,  -0.56172234, -0.35611769,  -0.35611769,  -0.20560465,  -0.20560465, 0.        ,   0.        ,   0.        ,   0.20560465, 0.20560465,   0.35611769,   0.35611769,   0.56172234, 0.56172234,   0.76732699,   0.76732699};

//x = {-2.0560465,  2.0560465, -5.6172234,  5.6172234, -2.0560465, 2.0560465, -7.6732699,  7.6732699, -4.1120929,  0.       , 4.1120929, -7.6732699,  7.6732699, -2.0560465,  2.0560465, -5.6172234,  5.6172234, -2.0560465,  2.0560465};
//z = {-7.6732699, -7.6732699, -5.6172234, -5.6172234, -3.5611769, -3.5611769, -2.0560465, -2.0560465,  0.       ,  0.       ,  0.       ,  2.0560465,  2.0560465,  3.5611769,  3.5611769, 5.6172234,  5.6172234,  7.6732699,  7.6732699};

// *5
// x = {  -1.028023 ,  1.028023 , -2.8086115,  2.8086115, -1.028023 , 1.028023 , -3.836635 ,  3.836635 , -2.0560465,  0.       ,  2.0560465, -3.836635 ,  3.836635 , -1.028023 ,  1.028023 ,    -2.8086115,  2.8086115, -1.028023 ,  1.028023	};
// z = {-3.83663495, -3.83663495, -2.8086117 , -2.8086117 , -1.78058845,       -1.78058845, -1.02802325, -1.02802325,  0.        ,  0.        ,        0.        ,  1.02802325,  1.02802325,  1.78058845,  1.78058845,        2.8086117 ,  2.8086117 ,  3.83663495,  3.83663495};

// *3
//x = {-0.6168138,  0.6168138, -1.6851669,  1.6851669, -0.6168138,        0.6168138, -2.301981 ,  2.301981 , -1.2336279,  0.       ,        1.2336279, -2.301981 ,  2.301981 , -0.6168138,  0.6168138,       -1.6851669,  1.6851669, -0.6168138,  0.6168138};
//z = {-2.30198097, -2.30198097, -1.68516702, -1.68516702, -1.06835307,        -1.06835307, -0.61681395, -0.61681395,  0.        ,  0.        ,        0.        ,  0.61681395,  0.61681395,  1.06835307,  1.06835307,        1.68516702,  1.68516702,  2.30198097,  2.30198097};

// *2
//x = {-0.4112092,  0.4112092, -1.1234446,  1.1234446, -0.4112092, 0.4112092, -1.534654 ,  1.534654 , -0.8224186,  0.       , 0.8224186, -1.534654 ,  1.534654 , -0.4112092,  0.4112092, -1.1234446,  1.1234446, -0.4112092,  0.4112092} ;
//z = {-1.53465398, -1.53465398, -1.12344468, -1.12344468, -0.71223538,       -0.71223538, -0.4112093 , -0.4112093 ,  0.        ,  0.        ,        0.        ,  0.4112093 ,  0.4112093 ,  0.71223538,  0.71223538,        1.12344468,  1.12344468,  1.53465398,  1.53465398};

// *2.2
x={ -0.45233012,  0.45233012, -1.23578906,  1.23578906, -0.45233012,        0.45233012, -1.6881194 ,  1.6881194 , -0.90466046,  0.        ,        0.90466046, -1.6881194 ,  1.6881194 , -0.45233012,  0.45233012,       -1.23578906,  1.23578906, -0.45233012,  0.45233012};
z = {-1.68811938, -1.68811938, -1.23578915, -1.23578915, -0.78345892,       -0.78345892, -0.45233023, -0.45233023,  0.        ,  0.        ,        0.        ,  0.45233023,  0.45233023,  0.78345892,  0.78345892,        1.23578915,  1.23578915,  1.68811938,  1.68811938};

count = -1;
For i In {1:NumMagnets}
	//z=z+1;

	// If((i)%6 == 0)
	// 	x=x+1; 
	// 	z=0;
	// EndIf
	count = count +1;
	// If(count == 10)
	//	height  = 10 * mm;
	// EndIf

	DefineConstant[
		X~{i} = {x(count)*45*mm, Min -100*mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
		Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
		Z~{i} = {z(count)*45*mm, Min -100*mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

		M~{i} = {0, Choices{0="Cylinder",1="Cube"},
		  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

		R~{i} = {radius, Min mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/1Radius [m]", i),
		  Visible (M~{i} == 0) },
		L~{i} = {height, Min mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/1Length [m]", i),
		  Visible (M~{i} == 0) },

		Lx~{i} = {50*mm, Min mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
		  Visible (M~{i} == 1) },
		Ly~{i} = {50*mm, Min mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
		  Visible (M~{i} == 1) },
		Lz~{i} = {50*mm, Min mm, Max 100*mm, Step mm,
		  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
		  Visible (M~{i} == 1) },

		Rx~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
		  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
		Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
		  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
		Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
		  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
	];
	
EndFor
