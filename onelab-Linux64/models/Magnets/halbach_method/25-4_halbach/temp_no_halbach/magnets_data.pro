DefineConstant[
  NumMagnets = {5, Min 1, Max 50, Step 1, Name "Parameters/0Number of magnets"}
];
mm = 1.e-3;

y = 0;


// MY MAGNET DIAMETER AND HEIGHT ARE 9*3MM * 19 of them
radius = 0.0045; 
height = 0.003;

// my cube magnet dimensions
/*
// >>> math.pow(math.pi * 0.0045 ** 2 * 0.003 * 19 / 21,1/3)
// 0.005568567483130031

side =  0.005568567483130031;
length_x = side;
length_y = side;
length_z = side;
spacing = 0.000252208799999998;

x = {	0,	length_x + spacing, (length_x + spacing)*2, (length_x + spacing)*3, (length_x + spacing)*4, 
		0,	 (length_x + spacing)*2,  (length_x + spacing)*4, 
		0,	length_x + spacing, (length_x + spacing)*2, (length_x + spacing)*3, (length_x + spacing)*4, 
		0,	 (length_x + spacing)*2,   (length_x + spacing)*4,
		0,	length_x + spacing, (length_x + spacing)*2, (length_x + spacing)*3,  (length_x + spacing)*4};

z = {	0,					0, 					0, 					0, 				0,
		length_z+ spacing,  length_z + spacing, length_z + spacing,
		(length_z + spacing)*2, (length_z + spacing)*2, (length_z + spacing)*2, (length_z + spacing)*2, (length_z + spacing)*2,
		(length_z + spacing)*3,  (length_z + spacing)*3,  (length_z + spacing)*3,
		(length_z + spacing)*4, (length_z + spacing)*4, (length_z + spacing)*4, (length_z + spacing)*4, (length_z + spacing)*4};
*/

// my cube magnet dimensions
// math.pow(math.pi * 0.0045 ** 2 * 0.003 * 19 / 5,1/3)
// 0.008984486294602786

side =  0.008984486294602786;
length_x = side;
length_y = side;
length_z = side;
spacing = 0.000252208799999998;

x = {	0,	length_x + spacing, (length_x + spacing)*2, (length_x + spacing)*3, (length_x + spacing)*4 };

z = {	0,					0, 					0, 					0, 				0};

count = -1;
For i In {1:NumMagnets}

	count = count +1;



		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },


			// Pi <=> 180 deg
			Rx~{i} = {Pi, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	


/*



	// NORTH UP
	If((count + 1) == 4)
	// If((count + 1) == 1 || (count + 1) == 5 || (count + 1) == 11 || (count + 1) == 17 || (count + 1) == 21)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },


			// Pi <=> 180 deg
			Rx~{i} = {Pi, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf


	// SOUTH UP
	If( (count + 1) == 2)
	// If( (count + 1) == 3 || (count + 1) == 9 || (count + 1) == 13 || (count + 1) == 19)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },

			Rx~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf

	// SOUTH FACING RIGHT
	If((count + 1) == 1 || (count + 1) == 5 )
	// If((count + 1) == 2 || (count + 1) == 12 || (count + 1) == 18)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },

			Rx~{i} = {Pi, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {Pi*3/2, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf


	// SOUTH FACING LEFT
	If((count + 1) == 3)
	// If((count + 1) == 4 || (count + 1) == 10 || (count + 1) == 20)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },

			Rx~{i} = {Pi, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {Pi/2, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf


	*/
	
	/*
	// SOUTH FACING FRONT
	// If((count + 1) == 7 || (count + 1) == 14 || (count + 1) == 16)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },

			Rx~{i} = {3*Pi/2, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf

	// SOUTH FACING BACK
	// If((count + 1) == 6 || (count + 1) == 8 || (count + 1) == 15)
		DefineConstant[
			X~{i} = {x(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
			Y~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
			Z~{i} = {z(count), Min -100*mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/0Z position [m]", i) },

			M~{i} = {1, Choices{0="Cylinder",1="Cube"},
			  Name Sprintf("Parameters/Magnet %g/00Shape", i)},

			Lx~{i} = {length_x, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1X length [m]", i),
			  Visible (M~{i} == 1) },
			Ly~{i} = {length_y, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Y length [m]", i),
			  Visible (M~{i} == 1) },
			Lz~{i} = {length_z, Min mm, Max 100*mm, Step mm,
			  Name Sprintf("Parameters/Magnet %g/1Z length [m]", i),
			  Visible (M~{i} == 1) },

			Rx~{i} = {Pi/2, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
			Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
			Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
			  Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
		];
	EndIf
	*/

EndFor
