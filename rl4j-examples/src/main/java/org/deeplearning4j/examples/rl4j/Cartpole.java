package org.deeplearning4j.examples.rl4j;


import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import java.util.logging.Logger;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * Cartpole DQN 메인 예제
 *
 * **/
public class Cartpole
{

    public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,    //랜덤 시드
                    200,    //에포크 최대 스텝
                    150000, //최대 스텝
                    150000, //반복 수행 최대 크기
                    32,     //배치 크기
                    500,    //타겟 업데이트 (hard)
                    10,     //noop 시작 스텝 수
                    0.01,   //보상 스케일링
                    0.99,   //gamma
                    1.0,    //td-error 클립핑
                    0.1f,   //최소 입실론
                    1000,   //입실론 탐욕 담금질(eps greedy anneal)을 위한 스텝 수
                    true    //이중 DQN
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET =
            new DQNFactoryStdDense.Configuration(
                    3,         //계층 수
                    16,        //은닉 노드 수
                    0.001,     //학습률
                    0.00       //l2 정규화
            );

    public static void main( String[] args )
    {
        cartPole();
        loadCartpole();
    }

    public static void cartPole() {

        //새 폴더의 rl4j-data에 학습 데이터 기록 (저장)
        DataManager manager = new DataManager(true);

        ///gym (name, render)을 이용해서 mdp 정의 
        GymEnv<Box, Integer, DiscreteSpace> mdp = null;
        try {
            mdp = new GymEnv("CartPole-v0", false, false);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }
        //학습 정의 
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);

        //학습
        dql.train();

        //최종 정책  
        DQNPolicy<Box> pol = dql.getPolicy();

        //직렬화와 저장 (직렬화는 필수는 아님)
        pol.save("/tmp/pol1");

        //mdp 종료 (http 연결)
        mdp.close();


    }


    public static void loadCartpole(){

        //유사한 새로운 mdp에서 학습된 에이전트를 이용하여 직렬화를 보여줌(렌더링은 이번에 수행)  

        //gym (name, render)을 이용해서 mdp 정의 
        GymEnv mdp2 = new GymEnv("CartPole-v0", true, false);

        //이전 에이전트 로드 
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");

        //에이전트 평가 
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }
}
