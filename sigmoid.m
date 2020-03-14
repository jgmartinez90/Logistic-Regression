function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));


%sigmoid function 
%remember .^ iterates to every element
g =(1 + exp(-1 * z)) .^ -1;


end
