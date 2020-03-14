function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


%hypothesis
h_sig = sigmoid(X * theta);
%regularization
reg = lambda / (2 * m) * (theta' * theta - theta(1)^2);
%cost function with regularization
J = 1 / m * (-y' * log(h_sig) - (1 - y') * log(1 - h_sig)) + reg;

mask = ones(size(theta));
mask(1) = 0;
%gradient
grad = 1 / m * X' * (h_sig - y) + lambda / m * (theta .* mask);

end
