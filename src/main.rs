use std::collections::HashMap;
// Initialize action-value function (Q) as dict

fn main() {

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

        
    // Example usage
    let state = 1; // Example state
    let q_values = get_q_values(state, action_space_size);

    println!("{:?}", q_values); // Outputs: [0.0, 0.0, 0.0, 0.0]
}
