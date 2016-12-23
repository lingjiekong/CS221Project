%% load the data file
MDP = load('MDP.txt');
C10 = load('qLearningWithControls10_400.txt');
C50 = load('qLearningWithControls50_400.txt');
C100 = load('qLearningWithControls100_400.txt');
COnly = load('Controller.txt');
QOnly = load('qLearningWithoutWarmStart.txt');
Tanh = load('qLearningWithTanh_400.txt');
Sigmoid = load('qlearningWithSigmoid_400.txt');
Relu = load('qLearningWithRelu_400.txt');
Elu = load('qLearningWithElu_400.txt');
COnlyElevation = load('controlsWithElevation400.txt');
C50Elevation = load('qLearningWithControlsWithElevation.txt');
Hidden1 = load('qLearning400_1hidden.txt');
Hidden10 = load('qLearning400_10hidden.txt');
Hidden100 = load('qLearning100Hidden_400.txt');

% define parameters
itr = 400;
smooth = 10;

% smooth data zero matrix
smoothCOnly = zeros(itr-smooth,1);
smoothQOnly = zeros(itr-smooth,1);
smoothMDP = zeros(itr-smooth,1);
smoothC10 = zeros(itr-smooth,1);
smoothC50 = zeros(itr-smooth,1);
smoothC100 = zeros(itr-smooth,1);
smoothTanh = zeros(itr-smooth,1);
smoothSigmoid = zeros(itr-smooth,1);
smoothRelu = zeros(itr-smooth,1);
smoothElu = zeros(itr-smooth,1);
smoothCOnlyElevation = zeros(itr-smooth,1);
smoothC50Elevation = zeros(itr-smooth,1);
smoothHidden1 = zeros(itr-smooth,1);
smoothHidden10 = zeros(itr-smooth,1);
smoothHidden100 = zeros(itr-smooth,1);


% smooth data
for i = 1:itr-smooth
    smoothCOnly(i) = sum(COnly(i:i+smooth))/smooth;
    smoothQOnly(i) = sum(QOnly(i:i+smooth))/smooth;
    smoothMDP(i) = sum(MDP(i:i+smooth))/smooth;
    smoothC10(i) = sum(C10(i:i+smooth))/smooth;
    smoothC50(i) = sum(C50(i:i+smooth))/smooth;
    smoothC100(i) = sum(C100(i:i+smooth))/smooth;
    smoothTanh(i) = sum(Tanh(i:i+smooth))/smooth;
    smoothSigmoid(i) = sum(Sigmoid(i:i+smooth))/smooth;
    smoothRelu(i) = sum(Relu(i:i+smooth))/smooth;
    smoothElu(i) = sum(Elu(i:i+smooth))/smooth;
    smoothCOnlyElevation(i) = sum(COnlyElevation(i:i+smooth))/smooth;
    smoothC50Elevation(i) = sum(C50Elevation(i:i+smooth))/smooth;
    smoothHidden1(i) = sum(Hidden1(i:i+smooth))/smooth;
    smoothHidden10(i) = sum(Hidden10(i:i+smooth))/smooth;
    smoothHidden100(i) = sum(Hidden100(i:i+smooth))/smooth;
end 

%% plot 1 for comparing MDP, Q-learning, and Controller
figure(1)
plot(smoothCOnly,'r')
hold on
plot(smoothMDP,'b')
hold on
plot(smoothQOnly,'k')
xlim([0,400]);
ylim([-400,350]);
legend('PD controller', 'MDP', 'Q-Learning', 'Location','southeast')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')
%% plot 2 for comparing different warm start
figure(2)
plot(smoothQOnly,'r')
hold on
plot(smoothC10,'b')
hold on
plot(smoothC50,'k')
hold on
plot(smoothC100,'g')
xlim([0,400]);
ylim([-400,350]);
legend('No Warm Start', '10 Itr', '50 Itr', '100 Itr', 'Location','southeast')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')
%% plot 3 for comparing different activation function
figure(3)
plot(smoothTanh,'r')
hold on
plot(smoothSigmoid,'b')
hold on
plot(smoothRelu,'k')
hold on
plot(smoothElu,'g') 
xlim([0,400]);
ylim([-400,350]);
legend('Tanh', 'Sigmoid', 'Relu', 'Elu', 'Location','southeast')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')
%% plot 4 different size of hidden layer
figure(4)
plot(smoothHidden1,'r')
hold on
plot(smoothHidden10,'b')
hold on
plot(smoothHidden100,'k')
xlim([0,400]);
ylim([-600,250]);
legend('1 Hidden Layer', '10 Hidden Layer', '100 Hidden Layer', 'Location','southeast')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')
%% plot 5 different environment
figure(5)
plot(smoothCOnly,'r')
hold on
plot(smoothC50,'b')
hold on
plot(smoothCOnlyElevation,'k')
hold on
plot(smoothC50Elevation,'g') 
xlim([0,400]);
ylim([-300,350]);
legend('Easy Environment Controller', 'Easy Environment Q-learing', ...
    'Hard Environment Controller', 'Hard Environment Q-learing', 'Location','southeast')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')
