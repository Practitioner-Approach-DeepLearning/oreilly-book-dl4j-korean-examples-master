package org.deeplearning4j.examples.rl4j;

import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;

import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * cartpole 문제에서 비동기 NStep Q러닝 실행 예제 
 */
public class AsyncNStepCartpole {


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration CARTPOLE_NSTEP =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,     //랜덤 시드
                    200,     //에포크 최대 스텝 
                    300000,  //최대 스텝 
                    16,      //쓰레드 수
                    5,       //t_max
                    100,     //타겟 업데이트(hard)
                    10,      //noop 시작 스텝 수
                    0.01,    //보상 스케일링
                    0.99,    //gamma
                    100.0,   //td-error 클립핑
                    0.1f,    //최소 입실론
                    9000     //입실론 탐욕 담금질(eps greedy anneal)을 위한 스텝 수
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET_NSTEP =
            new DQNFactoryStdDense.Configuration(
                    3,         //계층 수
                    16,        //은닉 노드 수
                    0.0005,    //학습률
                    0.001      //l2 정규화
            );


    public static void main( String[] args )
    {
        cartPole();
    }


    public static void cartPole() {

        //새 폴더의 rl4j-data에 학습 데이터 기록
        DataManager manager = new DataManager(true);

        //gym (name, render)을 이용해서 mdp 정의 
        GymEnv mdp = null;
        try {
        mdp = new GymEnv("CartPole-v0", false, false);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }

        //학습 정의 
        AsyncNStepQLearningDiscreteDense<Box> dql = new AsyncNStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET_NSTEP, CARTPOLE_NSTEP, manager);

        //학습
        dql.train();

        //mdp 종료 (http 연결)
        mdp.close();


    }


}
