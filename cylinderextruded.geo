lc0 = 0.25;
lc1 = 0.125;

width = 4.5;
length = 25.0;
height = 2.5;
radius = 0.5;

Point(1) = {0, 0, 0, lc0};
Point(2) = {length, 0,  0, lc0};
Point(3) = {length, width, 0, lc0};
Point(4) = {0,  width, 0, lc0};

cylinder_x0 = 0.1*length;
cylinder_y0 = width/2.0;
Point(5) = {cylinder_x0, cylinder_y0, 0.0, lc1};
Point(6) = {cylinder_x0 - radius, cylinder_y0, 0.0, lc1};
Point(7) = {cylinder_x0, cylinder_y0 + radius, 0.0, lc1};
Point(8) = {cylinder_x0 + radius, cylinder_y0, 0.0, lc1};
Point(9) = {cylinder_x0, cylinder_y0 - radius, 0.0, lc1};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Line Loop(5) = {1, 2, 3, 4};
Line Loop(6) = {5, 6, 7, 8};

Plane Surface(7) = {5, 6};

Recombine Surface {7};
Mesh.Algorithm = 8;

Extrude {0, 0, height}
{
  Surface{7}; Layers{height/lc0}; Recombine;
}

// The only way to determine the extruded surface numbers is to inspect GMSH's
// results. This may vary with newer version of GMSH.

// no-slip edges
Physical Surface(1) = {21, 7, 29, 50};
// inflow
Physical Surface(2) = {33};
// outflow
Physical Surface(3) = {25};
// cylinder
Physical Surface(4) = {37, 41, 45, 49};
// interior
Physical Volume(8) = {1};
