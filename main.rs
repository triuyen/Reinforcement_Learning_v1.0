use std::collections::HashMap;
use rand::Rng;
use rgym::
// Initialize action-value function (Q) as dict

mod line_world_env{
    pub fn run(num_case: &mut i32){
        // If there is no pre-determined num_case than put in random number between 5 and 10
        if !num_case{
            let mut rand_num = rand::thread_rng();
            let mut random_number: i32;

            loop {
                random_number = rand_num.gen_range(5..11);
                if random_number % 2 != 0{
                    break;
                }
            }
            
            
        };

        // the character always spawns in the middle

    };   
};

fn main() {

    let env = 

    // variable ajustable
    let  epsilon = 0.1; // epsilon greedy-policy
    let  gamma = 0.9 ;   // discount factor
    let  alpha = 0.02  ; // learning rate

    let  num_episodes = 50000 ; // number of episodes to run
    let  eval_interval = 1000;

    // Define the number of actions (like env.action_space.n in Python)
    let action_space_size = 4; // Example: say there are 4 possible actions

    // Create the Q table: a HashMap that maps a state (i32) to a Vec of zeros (for actions)
    let mut q_table: HashMap<i32, Vec<f64>> = HashMap::new();

    // Function to get or initialize the Q-values for a given state
    let mut get_q_values = |state: i32, action_space_size: usize| -> Vec<f64> 
        {
            q_table
                .entry(state)
                .or_insert_with(|| vec![0.0; action_space_size])
                .clone() // return a clone of the vector of Q-values
        };
    
    fn epsilon_greedy_policy(state, epsilon) {
            let mut rand_num = rand::thread_rng();

            if  let random_number: i32 = rand_num.gen_range(0..1) < epsilon{
                    return env.action_space.sample()
            };
        };

    // Example usage
    let state = 1; // Example state
    let q_values = get_q_values(state, action_space_size);

    println!("{:?}", q_values); // Outputs: [0.0, 0.0, 0.0, 0.0]
}
