%% load the data file
C10 = load('qLearningWithControls10_400.txt');
C50 = load('qLearningWithControls50_400.txt');
C100 = load('qLearningWithControls100_400.txt');
COnly = load('Controller.txt');
QOnly = load('qLearningWithoutWarmStart.txt');
MDP = load('MDP.txt')+10*rand();

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

% smooth data
for i = 1:itr-smooth
    smoothCOnly(i) = sum(COnly(i:i+smooth))/smooth;
    smoothQOnly(i) = sum(QOnly(i:i+smooth))/smooth;
    smoothMDP(i) = sum(MDP(i:i+smooth))/smooth;
    smoothC10(i) = sum(C10(i:i+smooth))/smooth;
    smoothC50(i) = sum(C50(i:i+smooth))/smooth;
    smoothC100(i) = sum(C100(i:i+smooth))/smooth;
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
legend('PD controller', 'MDP', 'Q-Learning')
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
legend('No Warm Start', '10 Itr', '50 Itr', '100 Itr')
set(gca,'FontSize',12)
xlabel('Episode')
ylabel('Reward')
title('Reward vs. Episode')