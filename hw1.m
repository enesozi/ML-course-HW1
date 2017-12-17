# Author: Enes Özipek
# Date: 13/03/2017

function [] = hw1()
  close all
  # Sample, order and feature number
  sampleN = 100;
  featureN = 20;
  orderMax = 7;
  # Init Y
  Y = zeros(sampleN,featureN);
  # Init error matrix
  err = normrnd(0,1,[sampleN,featureN]);
  # Init x vector
  x = sort(unifrnd(0,5,[1,featureN]));
  # Calculate actual result
  actualY = 2*sin(1.5*x);
  # Y = y + err
  for i =1:sampleN
    Y(i,:) = actualY+ err(i,:);
  end
  
  # Hipotesis values kept according to order values
  hips = {};
    
  for i =1:sampleN
    #scatter(x,Y(i,:))
    #hold on;
    h = zeros(orderMax,featureN);
     for j=1:orderMax
        D = genPolyMatrix(x.',j);
        w = fliplr((inv((D.')*D)*(D.')*(Y(i,:).')).');
        y1 = customPolyVal(w,x);
        hips{j}(i,:) = y1;
     end 
  end
  #bias and variance for orders
  biasSquared = zeros(1,orderMax);
  variance = zeros(1,orderMax);
  #Calculate bias,variance,and error
  for order = 1:orderMax
    biasSquared(order) = mean((mean(hips{order})-actualY).^2);
    variance(order) = mean(var(hips{order},1));
  end
  #Plot bias and variance plot
  figure
  hold on;
  plot(biasSquared,'k','Linewidth',2);
  plot(variance,'b','Linewidth',2);
  plot(biasSquared + variance,'g--.','Linewidth',2);
  xlabel('Polynomial Order');
  legend('Bias^2','Variance','Bias^2+Var.');
  hold off;
    
    # Trained data number
    nTr = ceil(featureN*.8);
    
    % CREATE TRAINING SET
    xTr = x(1:nTr);
    yTr = Y(:,1:nTr);
   
 
    % CREATE TESTING SET
    xTest = x(nTr+1:end);
    
    # Actual Y Training and Testing
    yTrAct = actualY(1:nTr);
    yTestAct = actualY(nTr+1:end);
    
    # Init cell arrays
    trainingHips = {};
    validHips = {}; 
    
    for i =1:sampleN
       for j=1:orderMax
          D = genPolyMatrix(xTr.',j);
          w = fliplr((inv((D.')*D)*(D.')*(yTr(i,:).')).');
          trainingHips{j}(i,:) = customPolyVal(w,xTr);
          validHips{j}(i,:) = customPolyVal(w,xTest);
       end 
    end
    
  #bias and variance for orders
  trainError = zeros(1,orderMax);
  validError = zeros(1,orderMax);
  
  #Calculate bias and variance error
  for order = 1:orderMax
    trainError(order) = mean((mean(trainingHips{order})-yTrAct).^2);
    validError(order) = mean((mean(validHips{order})-yTestAct).^2);
  end
  #Plot bias and variance plot
  figure
  hold on;
  plot(trainError,'k','Linewidth',2);
  plot(validError,'b','Linewidth',2);
  xlabel('Polynomial Order');
  legend('Train Error','Validation Error');
  hold off;
  
  # Iris Data Import  
  A = importdata('iris-data.txt',',');
  Dmatrix = A.data(:,1:4);
  Tmatrix = A.textdata;
  # Size of the data 
  DataSize = numel(Tmatrix);
  # Ratio of training data
  trDataRate = .8;
  y = zeros(DataSize,1);
  y(find(strcmp(Tmatrix, 'Iris-setosa').' == 1)) = 1;
  # Training data extraction
  trainingIndex = randperm(DataSize,ceil(DataSize*trDataRate));
  traX = Dmatrix(trainingIndex,:);
  traY = y(trainingIndex,:);
  # Testing data extraction
  testingIndex = setdiff([1:1:DataSize],trainingIndex);
  testX = Dmatrix(testingIndex,:);
  testY = y(testingIndex,:); 
  #Estimating mu and sigma
  setosa = traX(find(traY==1),:);        % data for setosa
  versicolor = traX(find(traY==0),:);    % data for versicolor
  ms = mean(setosa); % mean of setoas
  mv = mean(versicolor); % mean of versicolor
  testSize = ceil((1-trDataRate)*DataSize); % test data number
  wrong = 0; 
  covs = cov(setosa); # Cov matrix for setosa class
  covv = cov(versicolor); # Cov matrix for versicolor class
  
  correct = 0;
  for k=1:testSize
    x = testX(k,:); % test sample
    probS = -0.5*((x-ms)*pinv(covs)*(x-ms).')-0.5*log(covs); % calculating ratio for comparing
    probV = -0.5*((x-mv)*pinv(covv)*(x-mv).')-0.5*log(covv); % calculating ratio for comparing
    c1 = mean(mean(probS));
    c2 = mean(mean(probV));
    # In this condition there is a huge difference btw c1 and c2.
    # My guess is based on this assumption
    if ((c1 >= c2) && (testY(k) == 1)) || ((c1 < c2) && (testY(k) == 0))
        correct = correct + 1;
    end    
  end
  printf("Correct guess ratio is %d%%\n",(correct/testSize)*100);
 
  # Following code for visualizing the features.
  # To correctly run the code, you should have the same number of samples from 
  # Setosa class and Versicolor class.
  
  #{
  Characteristics = {'sepal length','sepal width','petal length','petal width'};
  pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
  h = figure;
  for j = 1:6,
      x = pairs(j, 1);
      y = pairs(j, 2);
      subplot(2,3,j);
      plot([setosa(:,x) versicolor(:,x)],...
           [setosa(:,y) versicolor(:,y)], '.');
      xlabel(Characteristics{x},'FontSize',10);
      ylabel(Characteristics{y},'FontSize',10);
      h = legend ("setosa", "versicolor");
      set(h, 'Location', 'northoutside'); 
      set (h,'interpreter','latex', "fontsize", 5);
  end
  #}
  
  
end

# This function calculates and return 
# polynomial values of given parameters
function [y] = customPolyVal(p,x)
 y = zeros(1,numel(x));
 n = numel(p);
 for i=n:-1:1
  y = y + p(n+1-i)*x.^(i-1);
 end
end

# Constructing Vendermode matrix 
# according to order number
function [D] = genPolyMatrix(x,n)
  D = ones(numel(x),n+1);
  for i=2:(n+1)
    D(:,i) = (x).^(i-1);
  end
end
