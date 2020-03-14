function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% hypothesis sigmoid function
h_sig = sigmoid(X * theta);
% Cost function
J = 1 / m * (-y' * log(h_sig) - (1 - y') * log(1 - h_sig));
%gradient 
grad = 1 / m * ((h_sig - y)' * X)';


end
