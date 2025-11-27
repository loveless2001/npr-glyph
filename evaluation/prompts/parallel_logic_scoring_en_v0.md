# Language Model Parallel Logic Scoring Criteria

## Scoring Framework Overview
Total Score: 100 points, divided into three core dimensions

## Dimension 1: Plan Independence (30 points)

### 1.1 Plan Differentiation (10 points)
- **Excellent (9-10 points)**: Each plan adopts completely different methodologies with no overlap
- **Good (7-8 points)**: Each plan has obvious differences with slight overlap
- **Fair (5-6 points)**: Plans have partial overlap but are still distinguishable
- **Poor (3-4 points)**: Plans are highly similar with minimal differences
- **Very Poor (0-2 points)**: Plans are essentially identical or logically contradictory

### 1.2 Methodological Completeness (10 points)
- **Excellent (9-10 points)**: Each plan is a self-sufficient complete strategy
- **Good (7-8 points)**: Plans are basically complete with slight dependencies
- **Fair (5-6 points)**: Plans require supplementary information for execution
- **Poor (3-4 points)**: Plans are incomplete, lacking key elements
- **Very Poor (0-2 points)**: Plans are vague or unexecutable

### 1.3 Logical Consistency (10 points)
- **Excellent (9-10 points)**: Internal logic of each plan is completely consistent
- **Good (7-8 points)**: Internal logic is basically consistent
- **Fair (5-6 points)**: Slight logical inconsistencies exist
- **Poor (3-4 points)**: Obvious logical problems exist
- **Very Poor (0-2 points)**: Logic is chaotic or self-contradictory

## Dimension 2: Execution Step Parallelism (40 points)

### 2.1 Step Independent Execution Capability (10 points)
- **Excellent (9-10 points)**: All steps can be executed completely in parallel with no dependencies
- **Good (7-8 points)**: Most steps can be parallelized with few dependencies
- **Fair (5-6 points)**: Some steps can be parallelized with moderate dependencies
- **Poor (3-4 points)**: Steps have many dependencies with low parallelism
- **Very Poor (0-2 points)**: Steps must be executed serially

### 2.2 Step Completeness (10 points)
- **Excellent (9-10 points)**: Step design is comprehensive, covering all necessary execution aspects
- **Good (7-8 points)**: Steps are basically complete with few details needing supplementation
- **Fair (5-6 points)**: Steps require additional explanation to ensure feasibility
- **Poor (3-4 points)**: Steps have obvious omissions, difficult to execute independently
- **Very Poor (0-2 points)**: Steps are vague or inoperable

### 2.3 Step Logic (10 points)
- **Excellent (9-10 points)**: Steps connect naturally with complete front-to-back consistency
- **Good (7-8 points)**: Step logic is basically reasonable with slight inconsistencies
- **Fair (5-6 points)**: Some logical gaps exist but execution is still barely possible
- **Poor (3-4 points)**: Logical relationships between steps are chaotic, affecting feasibility
- **Very Poor (0-2 points)**: Serious logical errors, steps are unfeasible

### 2.4 Analysis Depth Consistency (10 points)
- **Excellent (9-10 points)**: Analysis depth of each step is equivalent with balanced quality
- **Good (7-8 points)**: Analysis depth is basically consistent
- **Fair (5-6 points)**: Some steps have shallow analysis
- **Poor (3-4 points)**: Analysis depth differences are obvious
- **Very Poor (0-2 points)**: Analysis depth is severely uneven

## Dimension 3: Comprehensive Analysis Quality (30 points)

### 3.1 Information Integration Capability (10 points)
- **Excellent (9-10 points)**: Perfect integration of results from all steps with no information loss
- **Good (7-8 points)**: Effective integration of most information
- **Fair (5-6 points)**: Key information integrated with some omissions
- **Poor (3-4 points)**: Information integration is insufficient
- **Very Poor (0-2 points)**: Lacks effective integration

### 3.2 Comparative Analysis Depth (10 points)
- **Excellent (9-10 points)**: In-depth comparison of pros and cons of each approach with thorough analysis
- **Good (7-8 points)**: Comparative analysis is relatively in-depth
- **Fair (5-6 points)**: Basic comparison conducted
- **Poor (3-4 points)**: Comparison is superficial
- **Very Poor (0-2 points)**: Lacks effective comparison

### 3.3 Decision Rationality (10 points)
- **Excellent (9-10 points)**: Provides optimal decision based on analysis with sufficient reasoning
- **Good (7-8 points)**: Decision is reasonable with adequate reasoning
- **Fair (5-6 points)**: Decision is acceptable with fair reasoning
- **Poor (3-4 points)**: Decision is poor with insufficient reasoning
- **Very Poor (0-2 points)**: Decision is inappropriate or lacks reasonable basis

## Rating Standards
- **Grade A (≥90 points)**: High parallelism, rigorous logic
- **Grade B (80-89 points)**: Good parallelism, clear logic
- **Grade C (70-79 points)**: Moderate parallelism, basically feasible
- **Grade D (60-69 points)**: Low parallelism, needs improvement
- **Grade E (<60 points)**: Insufficient parallel capability, not recommended for use