package org.deeplearning4j.examples.rl4j;

import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparate;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 *
 * A3C를 이용한 cartpole 문제 예제 
 *
 */
public class A3CCartpole {

    private static A3CDiscrete.A3CConfiguration CARTPOLE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //랜덤 시드
                    200,            //에포크 최대 스텝 
                    500000,         //최대 스텝 
                    16,              //쓰레드 수
                    5,              //t_max
                    10,             // noop 시작 스텝 수
                    0.01,           //보상 스케일링
                    0.99,           //gamma
                    10.0           //td-error 클립핑
            );



    private static final ActorCriticFactorySeparateStdDense.Configuration CARTPOLE_NET_A3C = new ActorCriticFactorySeparateStdDense.Configuration(
            3,                      //계층 수
            16,                     //은닉 노드 수
            0.001,                 //학습률
            0.000                   //l2 정규화 
    );


    public static void main( String[] args )
    {
        A3CcartPole();
    }

    public static void A3CcartPole() {

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
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<Box>(mdp, CARTPOLE_NET_A3C, CARTPOLE_A3C, manager);

        //학습 시작
        dql.train();

        //mdp 종료 (http 연결)
        mdp.close();

    }



}
