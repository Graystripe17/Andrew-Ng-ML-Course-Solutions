function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a_1 = X
bias_1 = ones(size(a_1, 1), 1);
a_1 = [bias_1, a_1]; % a_1 is X
z_2 = Theta1 * a_1';
a_2 = sigmoid(z_2)'; % htheta1 is now 5000x25

bias_2 = ones(size(a_2, 1), 1);
a_2 = [bias_2, a_2];
z_3 = Theta2 * a_2';
a_3 = sigmoid(z_3); % output layer

output = a_3

% y_encoded = reshape(y(:), 5000, 11)
% Normally size of output is 10 but in this case there's 4
y_encoded = zeros(size(output, 1), size(y, 1));
for i = 1:size(y, 1)
    y_encoded(y(i), i) = 1;
endfor
costs_matrix = (1 / m) * (-y_encoded .* log(output) - (1 - y_encoded) .* log(1 - output));
J = sum(sum(costs_matrix));

regularized = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + regularized;

% -------------------------------------------------------------
function ans = sigmoid_prime(z)
    ans = sigmoid(z) .* (1 - sigmoid(z))
end
% -------------------------------------------------------------
for t = 1:m % m = 16
delta_3 = (output - y_encoded)(:, t); % (4x1)
keyboard;
% Theta2' (5x4); delta_3 (4x1); z_2 (4x16)
delta_2 = Theta2(:, 2:end)' * delta_3 .* sigmoid_prime(z_2(:, t));
Theta2_grad = Theta2_grad + delta_3 * a_2(t, 2:end);
Theta1_grad = Theta1_grad + delta_2 * a_1(t, 2:end);
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
