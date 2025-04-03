use crate::contracts::model_free_env::ModelFreeEnv;
use rand::prelude::SliceRandom;
use rand::Rng;

pub fn dyna_q<TEnv: ModelFreeEnv>(
    num_episodes: usize,
    learning_rate: f32,
    gamma: f32,
    epsilon: f32,
    env_name: &str,
    sim_updates: Option<usize>,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    use std::fs;
    use std::io::Write;

    let dir_path = format!("data/{}/dyna_q", env_name);
    let avg_rewards_path = format!("{}/{}_dyna_q_averaged_rewards.csv", dir_path, env_name);
    let episode_rewards_path = format!("{}/{}_dyna_q_episode_rewards.csv", dir_path, env_name);
    let cumulative_rewards_path =
        format!("{}/{}_dyna_q_cumulative_rewards.csv", dir_path, env_name);
    fs::create_dir_all(&dir_path).unwrap();

    let mut avg_rewards_file = fs::File::create(avg_rewards_path).unwrap();
    let mut episode_rewards_file = fs::File::create(episode_rewards_path).unwrap();
    let mut cumulative_rewards_file = fs::File::create(cumulative_rewards_path).unwrap();

    writeln!(avg_rewards_file, "interval,average_reward").unwrap();
    writeln!(episode_rewards_file, "episode,reward").unwrap();
    writeln!(cumulative_rewards_file, "episode,cumulative_reward").unwrap();

    let mut q_values = vec![vec![0.0; TEnv::num_actions()]; TEnv::num_states()];
    let mut model = vec![vec![None; TEnv::num_actions()]; TEnv::num_states()];
    let mut experiences = Vec::new();
    let mut rng = rand::thread_rng();

    let mut total_reward_batch = 0.0;
    let mut cumulative_reward = 0.0;

    for episode in 0..num_episodes {
        let mut env = TEnv::new();
        let mut total_reward = 0.0;

        env.reset();
        while !env.is_game_over() {
            let s = env.state_id();
            let available_actions = env.available_actions();
            let a = if rng.gen::<f32>() < epsilon {
                *available_actions.choose(&mut rng).unwrap()
            } else {
                q_values[s]
                    .iter()
                    .enumerate()
                    .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
                    .unwrap()
                    .0
            };
            let previous_score = env.score();
            env.step(a);
            let r = env.score() - previous_score;
            total_reward += r;
            let s_p = env.state_id();

            let max_q_s_p = q_values[s_p]
                .iter()
                .max_by(|q1, q2| q1.partial_cmp(q2).unwrap())
                .unwrap();
            q_values[s][a] += learning_rate * (r + gamma * max_q_s_p - q_values[s][a]);

            model[s][a] = Some((s_p, r));
            experiences.push((s, a, s_p, r));

            for _ in 0..sim_updates.unwrap_or(10) {
                if let Some(&(s, a, s_p, r)) = experiences.choose(&mut rng) {
                    let max_q_s_p = q_values[s_p]
                        .iter()
                        .max_by(|q1, q2| q1.partial_cmp(q2).unwrap())
                        .unwrap();
                    q_values[s][a] += learning_rate * (r + gamma * max_q_s_p - q_values[s][a]);
                }
            }
        }

        cumulative_reward += total_reward;
        total_reward_batch += total_reward;

        writeln!(episode_rewards_file, "{},{}", episode, total_reward).unwrap();
        writeln!(cumulative_rewards_file, "{},{}", episode, cumulative_reward).unwrap();

        if (episode + 1) % 10_000 == 0 {
            let avg_reward = total_reward_batch / 10_000.0;
            writeln!(
                avg_rewards_file,
                "{},{}",
                (episode + 1) / 10_000,
                avg_reward
            )
            .unwrap();
            total_reward_batch = 0.0;
        }
    }

    let mut policy = vec![0usize; TEnv::num_states()];
    for s in 0..TEnv::num_states() {
        policy[s] = q_values[s]
            .iter()
            .enumerate()
            .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap())
            .unwrap()
            .0;
    }

    (q_values, policy)
}
