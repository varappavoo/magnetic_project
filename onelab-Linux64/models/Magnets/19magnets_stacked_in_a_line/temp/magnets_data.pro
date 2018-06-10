magnets_count = 19;

DefineConstant[
  NumMagnets = {magnets_count, Min 1, Max 100, Step 1, Name "Parameters/0Number of magnets"}
];
mm = 1.e-3;

// MY MAGNET DIAMETER AND HEIGHT ARE 9*3MM
radius = 0.0045; 
height = 0.003;

y=(magnets_count/2);
For i In {1:NumMagnets}
  
  // distribute magnets evenly over the positie and negative y axis 
  y = y - 1;

  DefineConstant[
    X~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Magnet %g/0X position [m]", i) },
    Y~{i} = { y*(height+0.00001), Min -100*mm, Max 100*mm, Step mm,
      Name Sprintf("Parameters/Magnet %g/0Y position [m]", i) },
    Z~{i} = {0, Min -100*mm, Max 100*mm, Step mm,
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

    Rx~{i} = {Pi, Min -Pi, Max Pi, Step Pi/180,
      Name Sprintf("Parameters/Magnet %g/2X rotation [rad]", i) },
    Ry~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
      Name Sprintf("Parameters/Magnet %g/2Y rotation [rad]", i) },
    Rz~{i} = {0, Min -Pi, Max Pi, Step Pi/180,
      Name Sprintf("Parameters/Magnet %g/2Z rotation [rad]", i) }
  ];
EndFor
