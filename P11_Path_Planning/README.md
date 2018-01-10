# Udacity, Self-Driving Car Engineer Nanodegree
# Term 3, Project 1 Path-Planning


# Acknowledgements:
- Udacity Term 3, lectures
- Udacity Term 3 simulator: https://github.com/udacity/self-driving-car-sim/releases).
- Udacity Term 3, Project 1, Project Repository: https://github.com/udacity/CarND-Path-Planning-Project 
- Spline function for C++:  http://kluge.in-chemnitz.de/opensource/spline/


# Project Summary:
In this project, I navigate a car around a circular highway. The challenges to accomplish this are:
- Driving as fast as possible but not faster than the speed limit of 50 mph
- Slowing down if vehicle ahead are going slowly
- Changing lanes if there are currently no other vehicles in those lanes
- Driving smoothly in the middle of a lane and also changing lanes smoothly
- Accelerating and slowing down smoothly
- Not leaving the highway lanes


# YouTube Video:
Video follows


# Compiling and running the code:

1. Go to the folder that contains the "src" folder (do not go into the "src" folder!)
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Note: cpp release compilation works; no need for debug info activation.


# Reflection on the program:

## lines 1-161: 
These are all the helper functions that were provided for solving the project. I did not change these functions, imports, and/or add anything. So this is still pretty much the "default" foundation that I am using. I am not going into detail of those. They are pretty straightforward after going through the course material.

## lines 167-203: default parameters and function for reading input data
Here are some variable declarations. The speed variable ref_vel is set to 0.0 in the beginning. It will be increases sequentially in order to have a smooth acceleration and to prevent jerk.

## lines 204-244:
This initializes some variables with the data input from the provided json-stream. However, no computations here yet.

## lines 246-248:
This declares a variable for each lane that tells the program whether it would be save to drive in that lane. The following code performs several tests and, if it finds a potential threat in a certain lane, sets the variable for that lane to false (e.g., left_lane_available=false). After all tests are done, the program can use the variables in order to define in which lane it shall be driving.

## lines 250-254:
If the vehicle is already driving in the leftmost or rightmost lane, then there is no lane available on the left/right anymore. While this appears intuitively, it reduces computations because no checks for those lanes need to be done.

## lines 262-313:
This is the computation-intensive part because the sensor_fusion data for all vehicles on the highway is analyzed. I am only using one loop over the sensor fusion vector in order to avoid unnecessary load. This works fine but means that I need to create and manage a few more variables than proposed in class. 

The program checks whether there is a vehicle ahead (40m distance), a vehicle in the left lane (50m distance ahead and 50m distance behind), and a vehicle in the right lane (50m distance ahead and 50m distance behind). If so, the accoring variables are set.

## lines 316-337:
Depending on the previous findings (are there vehicles in the three lanes? which lane is available?), the set of if-statements defines the action of the vehicle. In particular, if there is no vehicle ahead, then the vehicle accelerates up to 48 mph and continues driving in the current lane. I tested 49 mph and 49.5 mph. However, I noticed that in some situations the vehicle surpassed the 50 mph limit. Thus, I set a maximum speed of 48 mph.

If there is another vehicle ahead, then the vehicle checks whether the left lane is available and, if so, moves into the left lane. If the left lane is not available, it would try the same in the right lane. (assuming there are lanes to the left and right)

Finally, if there is another vehicle ahead and it is not save to move into another lane, then we would slow down and follow the vehicle driving in front of us.

## lines 340-342:
This declares double-vectors for x and y points that later will be interpolated with a spline and additional points in order to have a predicted points that will be visited every 0.02 seconds. 

## lines 346-348:
This declares and initializes x, y, and yaw (--> angle) variables. The initialization is only needed if there are already (x,y) points from a prior iteration.

## lines 352-383:
This computes the first two (x,y) points. If there are not at least two (x,y) points from a previous iteration (lines 352-364), then the current position of the car is used as starting reference point and previous point is copmuted. If there are at least two computed (x,y) points already (lines 365-383), then the last two points of the previous path's (i.e., the previous two end points) are used as starting reference point. 

## lines 385-396:
In addition to the two starting points, the program now uses Frenet to compute three additional (x,y) points that are 30m apart in the target lane.

## lines 299-409:
This for-loop iterates through the vector of copmuted (x,y) points. It transforms all points to a reference angle of 0 degree. (Previously the reference angle depended on the first (x,y) point in the vector.)

## lines 410-415:
This creates a spline along the computed (x,y) points.

## lines 417-419:
This declares two double-vectors that store the (x,y) points that the car shall visit sequentially every 0.02 seconds. Initially, these two vectors are initialized with the (x,y) points from the previous computation. 

## lines 428-454:
This predicts the next 50 (x,y) points that shall we will visit within 0.02 second intervals. These computed (x,y) points are then stored in the two double-vectors for the x and respectively the y points. Predicting 50 points is recommended in the lecture and I noticed that 50 works fine. Too few points may lead to sudden changes, while too many points may impede me from reacting to new situations on the highway. (In addition, too many points could reduce performance.)

In order for the points to be 0.02 seconds apart, I am using the formula from the lecture: N * 0.02 * target velocity = target distance; where N is the number of waypoints between the current position and the target position (line 439).

## lines 461-465:
This transforms the double-vectors of (x,y) points into JSON objects and outputs these objects.(They will be read by the simulator then.) 


# Reflection on current parameters and potential future optimizations:
The aforementioned parameters could be tuned. Currently, the vehicle is driving very cautiously in order to avoid collisions. This is good in many situations. However, sometimes it means that the car slows down too much if there are slower vehicles ahead even if those vehicles are already accelerating again.

Besides, the lane change can be optimized too. Currently, the vehicle only changes lanes if there is a 100m gap (50m in front of the vehicle + 50m behind the vehicle). Again, this is a very cautious implementation. Alternatively, the vehicle could adjust the required gap when analyzing the speed difference to the vehicles in other lanes.

Finally, the vehicle could distinguish between the different vehicles in certain lanes. Currently, it focuses on vehicles in lanes within a certain distance. However, theoretically there could be multiple vehicles within that distance if the simulated vehicles are driving bad themselves. In these situations, there could still be a potential threat of collisions if the vehicle believes that it is following the vehicle ahead and then finds out that there was still a vehicle in between. 


