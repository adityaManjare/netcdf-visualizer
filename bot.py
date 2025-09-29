import os
import subprocess
import sys
import json
from dotenv import load_dotenv
from groq import Groq

# --- NEW: Import plotting libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
load_dotenv()
VECTOR_SEARCH_SCRIPT = "find_similar.py"
MODEL_NAME = "llama-3.1-8b-instant"
# --- NEW: Define a directory for saving plots ---
PLOT_OUTPUT_DIR = "plots" 
# --- NEW: Define the data directory ---
DATA_DIR = "parquets"


def setup_and_validate():
    """Validates that the necessary script and API key are available."""
    if not os.path.exists(VECTOR_SEARCH_SCRIPT):
        print(f"❌ Error: The script '{VECTOR_SEARCH_SCRIPT}' was not found.", file=sys.stderr)
        print("   Please ensure both bot.py and find_similar.py are in the same folder.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"❌ Error: GROQ_API_KEY is not set in your .env file.", file=sys.stderr)
        print("   Please follow the instructions in README.md to get your key and create the .env file.", file=sys.stderr)
        sys.exit(1)
    
    # --- NEW: Create the plots directory if it doesn't exist ---
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    return api_key

def execute_vector_search(query: str, num_results: int = 1) -> dict:
    """Calls the find_similar.py script and returns parsed JSON results."""
    command = [sys.executable, VECTOR_SEARCH_SCRIPT, query, str(num_results)]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        
        try:
            search_data = json.loads(result.stdout)
            return search_data
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse search results: {e}",
                "raw_output": result.stdout
            }
            
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Search script failed: {e.stderr}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

# --- NEW: Plotting function based on your script ---
def plot_ocean_profile(csv_filename: str) -> str:
    """
    Generates and saves a plot for a given ocean profile CSV file.
    The plot contains subplots for Temperature, Salinity, and Pressure vs. Depth.
    """
    try:
        # Construct the full file path
        file_path = os.path.join(DATA_DIR, csv_filename)

        if not os.path.exists(file_path):
            return json.dumps({
                "success": False, 
                "error": f"File not found at '{file_path}'. Please verify the filename."
            })

        df = pd.read_csv(file_path)

        # Check for required columns
        required_cols = ['TEMP', 'PSAL', 'PRES', 'level']
        if not all(col in df.columns for col in required_cols):
             return json.dumps({
                "success": False, 
                "error": f"CSV file is missing one of the required columns: {required_cols}"
            })

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
        fig.suptitle(f'Oceanographic Profile Analysis for: {csv_filename}', fontsize=16)

        # Plot Temperature
        sns.lineplot(ax=axes[0], data=df, y='PRES', x='TEMP', marker='o', color='r')
        axes[0].set_title('Temperature vs. Pressure (Depth)')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Pressure (dbar) ≈ Depth (m)')

        # Plot Salinity
        sns.lineplot(ax=axes[1], data=df, y='PRES', x='PSAL', marker='o', color='g')
        axes[1].set_title('Salinity vs. Pressure (Depth)')
        axes[1].set_xlabel('Salinity (PSU)')
        axes[1].set_ylabel('')

        # Plot Level (as a reference)
        sns.lineplot(ax=axes[2], data=df, y='PRES', x='level', marker='o', color='b')
        axes[2].set_title('Measurement Level vs. Pressure (Depth)')
        axes[2].set_xlabel('Measurement Level')
        axes[2].set_ylabel('')

        # Invert y-axis to show depth increasing downwards
        for ax in axes:
            ax.invert_yaxis()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot to the plots directory
        output_filename = os.path.splitext(csv_filename)[0] + '.png'
        output_path = os.path.join(PLOT_OUTPUT_DIR, output_filename)
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free up memory

        print(f"✅ Plot saved to '{output_path}'", file=sys.stderr)
        
        return json.dumps({
            "success": True, 
            "message": f"Plot successfully generated and saved to '{output_path}'."
        })

    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to create plot: {str(e)}"})


def enrich_query_to_summary_format(client, user_query: str, model_name: str) -> str:
    """
    Enriches the user query to match the exact format of stored summaries.
    This dramatically improves vector search accuracy.
    """
    enrichment_prompt = f"""Transform this ocean search query to match the database summary format.

Database summaries follow this pattern:
"recorded in [Ocean] at coordinates [lat]°[N/S], [lon]°[E/W], within [climatic zone] waters. contains measurements, spanning minimum depth [X]m maximum [Y]m. water column characteristics [types]. temperature gradient maximum [max]°C minimum [min]°C average. salinity extremes minimum [min] PSU maximum [max] PSU average."

Query: "{user_query}"

OUTPUT ONLY THE ENRICHED QUERY - NO EXPLANATIONS!

Rules:
- Add ocean region (Indian Ocean, Pacific, Atlantic, Southern Ocean, Arctic)
- Add climatic zones (tropical waters, polar waters, subtropical, equatorial)
- Add relevant terms: recorded, coordinates, temperature extremes, salinity PSU, depth measurements, water column, characteristics
- Keep it concise and natural

Examples:
"Southern Ocean data" → "recorded in Southern Ocean polar waters temperature extremes salinity PSU depth measurements characteristics"
"tropical profiles" → "tropical waters warm water characteristics temperature maximum salinity measurements"
"Indian region 5 degrees latitude" → "recorded in Indian Ocean tropical waters coordinates equatorial characteristics temperature salinity"

Transform now (OUTPUT ONLY THE QUERY):"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": enrichment_prompt}],
            model=model_name,
            max_tokens=300,
            temperature=0.3
        )
        
        enriched = response.choices[0].message.content.strip()
        
        # Clean up the response - remove explanations
        if '\n' in enriched:
            # Take only the first line if multiple lines
            enriched = enriched.split('\n')[0].strip()
        
        # Remove common prefix patterns
        prefixes_to_remove = [
            "Output:", "Query:", "Enriched:", "Transformed:", 
            "Here's the enriched query:", "The enriched query is:"
        ]
        for prefix in prefixes_to_remove:
            if enriched.lower().startswith(prefix.lower()):
                enriched = enriched[len(prefix):].strip()
        
        # Remove quotes if present
        enriched = enriched.strip('"').strip("'")
        
        print(f"🔍 Query enrichment:", file=sys.stderr)
        print(f"   Original: '{user_query}'", file=sys.stderr)
        print(f"   Enriched: '{enriched}'", file=sys.stderr)
        return enriched
        
    except Exception as e:
        print(f"⚠️  Query enrichment failed: {e}, using original query", file=sys.stderr)
        return user_query

def ask_clarifying_questions(client, user_query: str, model_name: str) -> dict:
    """
    Analyzes query completeness and asks clarifying questions if too vague.
    Returns dict with 'needs_clarification' boolean and 'questions' or 'enriched_query'.
    """
    analysis_prompt = f"""Analyze this ocean profile search query: "{user_query}"

The database summaries contain:
- Ocean regions (Southern Ocean, Pacific, Atlantic, Indian, Arctic)
- Precise coordinates (latitude/longitude)
- Climatic zones (southern polar, tropical, subtropical, equatorial)
- Temperature extremes (min/max/average in °C)
- Salinity extremes (min/max/average in PSU)
- Depth ranges (minimum to maximum in meters)
- Water characteristics (cold water, warm water, deep waters, surface waters)

Determine if the query is:
1. SPECIFIC ENOUGH - has clear search criteria (coordinates, ocean region, specific conditions)
2. TOO VAGUE - needs clarification to get meaningful results

If SPECIFIC ENOUGH, respond with JSON:
{{"needs_clarification": false, "reasoning": "brief reason"}}

If TOO VAGUE, respond with JSON:
{{"needs_clarification": true, "questions": ["question 1", "question 2"]}}

Ask 1-2 brief questions about the MOST important missing information.

Examples:
- "ocean data" → TOO VAGUE (needs region, conditions, or coordinates)
- "Southern Ocean profiles" → SPECIFIC ENOUGH (has ocean region)
- "temperature data" → TOO VAGUE (needs location)
- "data at 60°S" → SPECIFIC ENOUGH (has coordinates)

Analyze: "{user_query}"
"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": analysis_prompt}],
            model=model_name,
            max_tokens=250,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_match = result_text
        if "```json" in result_text:
            json_match = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            json_match = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(json_match)
        return result
        
    except Exception as e:
        print(f"⚠️  Query analysis failed: {e}, proceeding without clarification", file=sys.stderr)
        return {"needs_clarification": False, "reasoning": "analysis failed"}

# --- MODIFIED: Added the new plot_ocean_profile tool ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_ocean_profiles",
            "description": "Search the vector database for the single best matching oceanographic profile. The database contains detailed summaries with ocean regions, coordinates, climatic zones, temperature/salinity extremes, depth ranges, and water characteristics. Always use this when users ask about ocean data, locations, or conditions. The search returns only the TOP 1 most relevant profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Will be automatically enriched to match the database summary format. Include any relevant: ocean regions, coordinates, temperature/salinity info, depth ranges, climatic zones, or water characteristics."
                    },
                    "enrich_query": {
                        "type": "boolean",
                        "description": "Whether to automatically enrich the query to match database format (default: true)",
                        "default": True
                    },
                    "ask_clarification": {
                        "type": "boolean", 
                        "description": "Whether to ask clarifying questions if query is vague (default: true)",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        # --- NEW: Definition for the plotting tool ---
        "type": "function",
        "function": {
            "name": "plot_ocean_profile",
            "description": "Generates and saves a plot of Temperature, Salinity, and Pressure vs. Depth from a specific ocean profile CSV file. Use this *after* a search has been performed to visualize the retrieved data. This tool should only be called if the user explicitly asks to 'plot', 'visualize', or 'graph' the data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_filename": {
                        "type": "string",
                        "description": "The name of the source CSV file (e.g., 'profile_0008.csv') obtained from the 'Source File' field of a previous search result."
                    }
                },
                "required": ["csv_filename"]
            }
        }
    }
]

def format_search_results(search_data: dict) -> str:
    """Format search results - optimized for single result."""
    if not search_data.get("success", False):
        return f"Search failed: {search_data.get('error', 'Unknown error')}"
    
    if search_data.get("num_results", 0) == 0:
        return f"No matching ocean profile found. {search_data.get('message', '')}"
    
    # Get the top result
    results = search_data.get("results", [])
    if not results:
        return "No results returned from search."
    
    profile = results[0]  # Only the top match
    
    formatted = f"**Best Matching Ocean Profile** (Similarity: {profile.get('similarity', 0):.4f})\n\n"
    formatted += f"**Profile ID:** {profile.get('profile_id', 'N/A')}\n"
    formatted += f"**Source File:** {profile.get('file', 'N/A')}\n" # This is the crucial filename
    
    lat = profile.get('latitude')
    lon = profile.get('longitude')
    if lat is not None and lon is not None:
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        formatted += f"**Location:** {abs(lat):.3f}°{lat_dir}, {abs(lon):.3f}°{lon_dir}\n\n"
    
    # Full summary
    formatted += f"**Complete Summary:**\n{profile.get('document', 'N/A')}\n"
    
    relevance = profile.get('relevance_score')
    if relevance:
        formatted += f"\n**Relevance Score:** {relevance}\n"
    
    return formatted.strip()

# --- MODIFIED: handle_tool_call now supports the new plotting function ---
def handle_tool_call(client, tool_call, model_name):
    """Execute the tool call with query enrichment and clarification."""
    function_name = tool_call.function.name
    
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return json.dumps({
            "success": False,
            "error": "Invalid function arguments"
        })

    if function_name == "search_ocean_profiles":
        query = arguments.get("query", "")
        enrich = arguments.get("enrich_query", True)
        ask_clarification_flag = arguments.get("ask_clarification", True)
        
        print(f"\n🔍 Processing search request for: '{query}'", file=sys.stderr)
        
        # Step 1: Check if clarification needed
        if ask_clarification_flag:
            clarification_result = ask_clarifying_questions(client, query, model_name)
            
            if clarification_result.get("needs_clarification"):
                questions = clarification_result.get("questions", [])
                return json.dumps({
                    "needs_clarification": True,
                    "questions": questions,
                    "message": "Query is too vague. Please provide more details."
                })
        
        # Step 2: Enrich the query
        if enrich:
            enriched_query = enrich_query_to_summary_format(client, query, model_name)
        else:
            enriched_query = query
        
        # Step 3: Execute search (only 1 result)
        print(f"🎯 Searching for TOP 1 result...", file=sys.stderr)
        search_data = execute_vector_search(enriched_query, num_results=1)
        
        # Format and return
        return format_search_results(search_data)
    
    # --- NEW: Handler for the plotting tool ---
    elif function_name == "plot_ocean_profile":
        csv_filename = arguments.get("csv_filename")
        if not csv_filename:
            return json.dumps({"success": False, "error": "csv_filename is required."})
        
        print(f"📈 Generating plot for '{csv_filename}'...", file=sys.stderr)
        return plot_ocean_profile(csv_filename)

    return json.dumps({"error": f"Unknown function: {function_name}"})


def main():
    """Main chat loop with function calling and query enrichment."""
    global MODEL_NAME
    
    groq_api_key = setup_and_validate()
    
    try:
        client = Groq(api_key=groq_api_key)
        print("✅ Groq client configured successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}", file=sys.stderr)
        sys.exit(1)

    # Test the model
    try:
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model=MODEL_NAME,
            max_tokens=10
        )
        print(f"\n✅ Using model '{MODEL_NAME}'.")
    except Exception as e:
        if "decommissioned" in str(e) or "not supported" in str(e):
            print(f"⚠️  Model '{MODEL_NAME}' is not available. Falling back to gemma2-9b-it")
            MODEL_NAME = "gemma2-9b-it"
        else:
            print(f"❌ Error testing model: {e}", file=sys.stderr)
            sys.exit(1)

    # --- MODIFIED: Updated system prompt to include instructions for plotting ---
    system_prompt = (
        "You are Argo Ocean Assistant, an expert oceanographer with access to a vector database "
        "of ocean profiles. Your capabilities include searching for data and plotting it.\n\n"
        "**Workflow:**\n"
        "1. **Search First:** When a user asks about ocean data, ALWAYS use the `search_ocean_profiles` function to find the single best matching profile.\n"
        "2. **Interpret Results:** After receiving search results, provide a detailed oceanographic interpretation focusing on the geographic/climatic context, temperature/salinity extremes, and scientific significance.\n"
        "3. **Plot on Request:** If, and only if, the user asks to 'plot', 'visualize', or 'graph' the data *after* a successful search, you MUST use the `plot_ocean_profile` function. Use the `Source File` value from the search result as the `csv_filename` argument for the plot tool.\n\n"
        "If a search is too vague, ask the user the suggested clarifying questions."
    )

    print(f"\n🌊 Argo Ocean Assistant (Search + Plotting)")
    print("   Ask about ocean conditions, then ask me to plot the results!")
    print("   (Type 'quit' or 'exit' to end)\n")

    messages = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("🤖 Goodbye! Happy oceanographic research!")
                break
            
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # Make API call with tools
            response = client.chat.completions.create(
                messages=messages,
                model=MODEL_NAME,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0.7
            )

            response_message = response.choices[0].message
            
            if response_message.tool_calls:
                messages.append(response_message)
                
                tool_call_results = []
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    print(f"🔧 Executing: {function_name}", file=sys.stderr)
                    
                    # Execute with enrichment
                    function_response = handle_tool_call(client, tool_call, MODEL_NAME)
                    
                    # Check if clarification needed from search tool
                    try:
                        response_data = json.loads(function_response)
                        if response_data.get("needs_clarification"):
                            questions = response_data.get("questions", [])
                            print(f"\n🤖 I need more details to search effectively:")
                            for i, q in enumerate(questions, 1):
                                print(f"   {i}. {q}")
                            print()
                            # Don't continue to LLM, wait for user response
                            messages.pop()  # Remove assistant message
                            messages.pop()  # Remove user message
                            continue
                    except (json.JSONDecodeError, TypeError):
                         # Not JSON or not a dict, so it's a normal response, continue
                        pass
                    
                    # Display results for search
                    if function_name == 'search_ocean_profiles':
                        print("\n" + "="*80)
                        print("📋 TOP MATCHING PROFILE:")
                        print("="*80)
                        print(function_response)
                        print("="*80 + "\n")
                    
                    tool_call_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })

                # If we had clarifying questions, the loop would have continued.
                # Now append all tool results.
                messages.extend(tool_call_results)
                
                # Get interpretation
                second_response = client.chat.completions.create(
                    messages=messages,
                    model=MODEL_NAME,
                    max_tokens=4096,
                    temperature=0.7
                )
                
                final_message = second_response.choices[0].message.content
                # Handle the output from the plotting tool differently
                if any(t["name"] == "plot_ocean_profile" for t in tool_call_results):
                     # The tool itself returns a user-friendly message, just print the LLM's confirmation
                    print(f"🤖 {final_message}\n")
                else:
                    print(f"🤖 INTERPRETATION:\n{final_message}\n")
                
                messages.append({"role": "assistant", "content": final_message})
                
            else:
                response_text = response_message.content
                print(f"🤖 {response_text}\n")
                messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n🤖 Goodbye! Happy oceanographic research!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()
            if messages and messages[-1]["role"] == "user":
                messages.pop()

if __name__ == "__main__":
    main()