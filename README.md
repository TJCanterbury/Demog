# Code for a dynamic game, investigating the role of feedback between population densities and life history influence the evolution of develpopmental windows

A detailed outline of how the code works and our motivation for making this model can be found in "\Pseudocode.pdf".

## Installation
The easiest way to simply run the model (if you  have a Windows) is to use the executable file "\target\debug\Demog1.exe".

Else, if you wish to play with the source code it can be found at "\src\main.rs".

To compile your edited code and run the model you will need to install [Rust](https://rust-lang.org/tools/install/). Cargo, Rust's package manner, will automatically install all the neccessary packages when you attempt to compile. You must include a project name as the first argument when you compile, e.g. see the argument "trial1" in the code below, this will output your results files to "\results\trial1\" 

```bash
git clone https://github.com/TJCanterbury/Demog.git
cd TheModel
cargo run trial1
```