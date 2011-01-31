clc
clear classes
clear

debug=0;

test_data = data_struct([[0 1]; [0 0]], [[1]; [1]]);

% Bug but the kernelized version gives the good result
test_case('Bug', data_struct([[0.5 1.5]; 
                              [1.5 0.5]], [[-1]; 
                                           [1]]), test_data);

% Mean not at the origin
test_case('Mean not at the origin', data_struct([[0 2]; 
                                                 [1 1]], [[-1]; 
                                                          [1]]), test_data);

% Simple example
%data =
test_case('Simple example',  data_struct([[0.5 1.5]; 
                                          [1.5 0.5]; 
                                          [0.5 1.5]; 
                                          [1.5 0.5]], [[-1]; 
                                                       [1]; 
                                                       [-1]; 
                                                       [1]]), test_data);


