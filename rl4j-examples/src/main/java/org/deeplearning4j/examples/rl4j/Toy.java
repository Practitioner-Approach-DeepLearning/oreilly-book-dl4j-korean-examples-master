package org.deeplearning4j.examples.rl4j;


import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * 토이 DQN 예제
 *
 */
public class Toy {


    public static QLearning.QLConfiguration TOY_QL =
            new QLearning.QLConfiguration(
                    123,   //랜덤 시드
                    100000,//에포크 최대 스텝 
                    80000, //최대 스텝
                    10000, //반복 수행 최대 크기
                    32,    //배치 크기
                    100,   //타겟 업데이트 (hard)
                    0,     //noop 시작 스텝 수
                    0.05,  //보상 스케일링
                    0.99,  //gamma
                    10.0,  //td-error 클립핑
                    0.1f,  //최소 입실론
                    2000,  //입실론 탐욕 담금질(eps greedy anneal)을 위한 스텝 수
                    true   //이중 DQN
            );


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration TOY_ASYNC_QL =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,        //랜덤 시드
                    100000,     //에포크 최대 스텝
                    80000,      //최대 스텝
                    8,          //쓰레드 수
                    5,          //t_max
                    100,        //타겟 업데이트 (hard)
                    0,          //noop 시작 스텝 수
                    0.1,        //보상 스케일링
                    0.99,       //gamma
                    10.0,       //td-error 클립핑
                    0.1f,       //최소 입실론
                    2000        //입실론 탐욕 담금질(eps greedy anneal)을 위한 스텝 수
            );

    public static DQNFactoryStdDense.Configuration TOY_NET =
            new DQNFactoryStdDense.Configuration(
                    3,        //계층 수
                    16,       //은닉 노드 수
                    0.001,    //학습률
                    0.01      //l2 정규화
            );

    public static void main(String[] args )
    {
        simpleToy();
        //toyAsyncNstep();

    }

    public static void simpleToy() {

        //새 폴더의 rl4j-data에 학습 데이터 기록.
        DataManager manager = new DataManager();

        //toy (toy length)로 부터 mdp 정의.
        SimpleToy mdp = new SimpleToy(20);

        //학습 메소드 정의.
        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QL, manager);

        //toy mdp에서 디버깅 목적으로 로깅 사용 가능.
        mdp.setFetchable(dql);

        //학습 시작.
        dql.train();

        //토이 예제에서는 필요 없지만 연습용.
        mdp.close();

    }

    public static void hardToy() {

        //새 폴더의 rl4j-data에 학습 데이터 기록 (저장)
        DataManager manager = new DataManager();

        //toy (toy length)로 부터 mdp 정의.
        MDP mdp = new HardDeteministicToy();

        //학습 정의.
        ILearning<SimpleToyState, Integer, DiscreteSpace> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);

        //학습 시작.
        dql.train();

        //토이 예제에서는 필요 없지만 연습용.
        mdp.close();


    }


    public static void toyAsyncNstep() {

        //새 폴더의 rl4j-data에 학습 데이터 기록 (저장)
        DataManager manager = new DataManager();

        //mdp 정의.
        SimpleToy mdp = new SimpleToy(20);

        //학습 정의. 
        AsyncNStepQLearningDiscreteDense dql = new AsyncNStepQLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_ASYNC_QL, manager);

        //toy mdp에서 디버깅 목적으로 로깅 사용 가능.
        mdp.setFetchable(dql);

        //학습 시작.
        dql.train();

        //토이 예제에서는 필요 없지만 연습용.
        mdp.close();

    }

}
