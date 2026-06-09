import subprocess
import os
import time
import typer
from google import genai
from google.genai import types

app = typer.Typer()
client = genai.Client()

SANDBOX_DIR = "sandbox"
PROTOTYPE_FILE = os.path.join(SANDBOX_DIR, "fenics_prototype.py")

def run_sandbox():
    print(f"🚀 Executing {PROTOTYPE_FILE}...")
    result = subprocess.run(
        ["python", "fenics_prototype.py"], # <-- We removed PROTOTYPE_FILE from this array
        capture_output=True, 
        text=True,
        cwd=SANDBOX_DIR
    )
    return result.returncode, result.stdout, result.stderr

def generate_with_retry(prompt, config, max_retries=4):
    """Wraps the API call with exponential backoff for 503/429 errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config
            )
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "Unavailable" in str(e):
                wait_time = (2 ** attempt) * 2  # Waits 2s, 4s, 8s, 16s
                print(f"⏳ Server busy. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e # Raise other unexpected errors
    print("❌ API is completely overloaded right now. Aborting.")
    return None

@app.command()
def build_prototype(max_iterations: int = 3):
    if not os.path.exists(SANDBOX_DIR):
        os.makedirs(SANDBOX_DIR)

    system_instruction = (
        "You are an expert computational mechanics engineer. "
        "Output ONLY valid, executable Python code inside a markdown block. "
        "Do not include explanations outside the code block."
    )
    
    prompt = (
        "Write a complete Python script. Do NOT solve a Poisson problem. "
        "Use these exact imports: `from dolfin import *` and `from dolfin_adjoint import *`. Do not use `import fenics_adjoint`. "
        "1. Define a 2D rectangular mesh (cantilever beam fixed on the left, point load on the right bottom corner). "
        "2. Define a linear elastic problem using a VectorFunctionSpace. "
        "3. Create a scalar Control variable 'p' (representing ALM laser power) in a FunctionSpace(mesh, 'CG', 1). "
        "4. Map 'p' to the Young's Modulus 'E' using a simple polynomial penalization: E = p**3 * E0. "
        "5. Solve the elasticity problem. "
        "6. Define the compliance as the objective function (J = assemble(inner(sigma, epsilon)*dx)). "
        "7. Compute the exact gradient of J with respect to 'p' using compute_gradient(J, Control(p)). "
        "8. Explicitly print the maximum value of the computed gradient array to the console. "
        "Do not plot the results. Just print the gradient."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1
    )

    print("🤖 Agent: Writing initial FEniCS code...")
    response = generate_with_retry(prompt, config)
    
    if not response:
        return

    code = response.text.replace("```python", "").replace("```", "").strip()
    with open(PROTOTYPE_FILE, "w") as f:
        f.write(code)

    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        returncode, stdout, stderr = run_sandbox()
        
        if returncode == 0:
            print("✅ Success! FEniCS solver completed and gradients computed.")
            print("Output:\n", stdout[:500])
            break
        else:
            print("❌ Solver failed. Error snapshot:\n" + stderr[:500] + "\n...\nFeeding to Debugger...")
            
            # UPDATE THIS PROMPT
            debug_prompt = (
                f"You are debugging a 2D linear elastic cantilever beam problem using FEniCS and dolfin-adjoint. "
                f"The goal is to map a control parameter 'p' to Young's Modulus and compute the gradient of compliance. "
                f"Use exact imports: `from dolfin import *` and `from dolfin_adjoint import *`. "
                f"The previous script failed with this error:\n\n{stderr}\n\n"
                f"Fix the mathematical or syntax bug. Do NOT change the problem to a Poisson equation. "
                f"Provide the complete, corrected Python code."
            )
            
            response = generate_with_retry(debug_prompt, config)
            if not response:
                break
                
            code = response.text.replace("```python", "").replace("```", "").strip()
            with open(PROTOTYPE_FILE, "w") as f:
                f.write(code)
    else:
        print("\n⚠️ Reached maximum iterations without converging.")

if __name__ == "__main__":
    app()