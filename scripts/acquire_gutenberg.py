"""
Acquire philosophy primary texts from Project Gutenberg.

Public domain. No crawling — just direct downloads of specific books.
Saves to corpus/raw/gutenberg/ as plain UTF-8 text.

Run: python scripts/acquire_gutenberg.py
"""

import os
import time
import requests
from pathlib import Path

# Curated philosophy texts. Each entry: (gutenberg_id, author, title, filename)
# IDs from gutenberg.org. Format URLs use the .txt.utf-8 version.
GUTENBERG_TEXTS = [
    # Plato
    (1497, "Plato", "The Republic", "plato_republic.txt"),
    (1656, "Plato", "Apology", "plato_apology.txt"),
    (1658, "Plato", "Phaedo", "plato_phaedo.txt"),
    (1635, "Plato", "Symposium", "plato_symposium.txt"),
    (1726, "Plato", "Phaedrus", "plato_phaedrus.txt"),
    (1750, "Plato", "Meno", "plato_meno.txt"),
    (1643, "Plato", "Crito", "plato_crito.txt"),
    
    # Aristotle
    (8438, "Aristotle", "Nicomachean Ethics", "aristotle_nicomachean_ethics.txt"),
    (6762, "Aristotle", "Politics", "aristotle_politics.txt"),
    (6763, "Aristotle", "Poetics", "aristotle_poetics.txt"),
    (59058, "Aristotle", "Metaphysics", "aristotle_metaphysics.txt"),
    
    # Hume
    (4705, "David Hume", "An Enquiry Concerning Human Understanding", "hume_enquiry.txt"),
    (9662, "David Hume", "A Treatise of Human Nature", "hume_treatise.txt"),
    
    # Kant
    (4280, "Immanuel Kant", "Critique of Pure Reason", "kant_critique_pure_reason.txt"),
    (5683, "Immanuel Kant", "Fundamental Principles of the Metaphysic of Morals", "kant_metaphysic_morals.txt"),
    (5684, "Immanuel Kant", "The Critique of Practical Reason", "kant_practical_reason.txt"),
    
    # Mill
    (11224, "John Stuart Mill", "Utilitarianism", "mill_utilitarianism.txt"),
    (34901, "John Stuart Mill", "On Liberty", "mill_on_liberty.txt"),
    
    # Nietzsche
    (1998, "Friedrich Nietzsche", "Thus Spake Zarathustra", "nietzsche_zarathustra.txt"),
    (4363, "Friedrich Nietzsche", "Beyond Good and Evil", "nietzsche_beyond_good_evil.txt"),
    (52263, "Friedrich Nietzsche", "The Genealogy of Morals", "nietzsche_genealogy_morals.txt"),
    
    # Descartes
    (59, "René Descartes", "Discourse on the Method", "descartes_discourse.txt"),
    (23306, "René Descartes", "Meditations on First Philosophy", "descartes_meditations.txt"),
    
    # Spinoza
    (3800, "Baruch Spinoza", "Ethics", "spinoza_ethics.txt"),
    
    # Locke
    (10615, "John Locke", "An Essay Concerning Human Understanding Vol 1", "locke_essay_v1.txt"),
    (10616, "John Locke", "An Essay Concerning Human Understanding Vol 2", "locke_essay_v2.txt"),
    (7370, "John Locke", "Second Treatise of Government", "locke_second_treatise.txt"),
    
    # Berkeley
    (4723, "George Berkeley", "A Treatise Concerning the Principles of Human Knowledge", "berkeley_principles.txt"),
    
    # Hobbes
    (3207, "Thomas Hobbes", "Leviathan", "hobbes_leviathan.txt"),
    
    # Rousseau
    (46333, "Jean-Jacques Rousseau", "The Social Contract", "rousseau_social_contract.txt"),
    
    # Schopenhauer
    (38427, "Arthur Schopenhauer", "The World as Will and Idea Vol 1", "schopenhauer_world_will_v1.txt"),
    
    # Marcus Aurelius
    (2680, "Marcus Aurelius", "Meditations", "marcus_aurelius_meditations.txt"),
    
    # Epictetus
    (45109, "Epictetus", "The Enchiridion", "epictetus_enchiridion.txt"),
    
    # Bacon
    (5500, "Francis Bacon", "Novum Organum", "bacon_novum_organum.txt"),
    
    # James
    (11984, "William James", "Pragmatism", "james_pragmatism.txt"),
    
    # Russell
    (5827, "Bertrand Russell", "The Problems of Philosophy", "russell_problems.txt"),
    (44932, "Bertrand Russell", "Mysticism and Logic", "russell_mysticism_logic.txt"),
]


def get_gutenberg_url(gutenberg_id: int) -> str:
    """Build the .txt.utf-8 URL for a Gutenberg ID."""
    return f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"


def download_text(gutenberg_id: int, filename: str, output_dir: Path, delay: float = 2.0) -> bool:
    """
    Download one Gutenberg text. Returns True on success.
    Skips if file already exists (resumable).
    """
    output_path = output_dir / filename
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  [skip] {filename} already exists ({output_path.stat().st_size:,} bytes)")
        return True
    
    url = get_gutenberg_url(gutenberg_id)
    try:
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "philosophy-llm/0.1 personal research project"
        })
        response.raise_for_status()
        
        # Gutenberg returns plain text. Save as UTF-8.
        content = response.content.decode("utf-8", errors="replace")
        
        if len(content) < 1000:
            print(f"  [warn] {filename} unexpectedly small ({len(content)} bytes), skipping save")
            return False
        
        output_path.write_text(content, encoding="utf-8")
        print(f"  [ok]   {filename} ({len(content):,} chars)")
        time.sleep(delay)  # be polite
        return True
        
    except Exception as e:
        print(f"  [fail] {filename} -- {e}")
        return False


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "corpus" / "raw" / "gutenberg"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Texts to acquire: {len(GUTENBERG_TEXTS)}")
    print(f"Polite delay: 2.0 seconds between requests\n")
    
    successes = 0
    failures = 0
    
    for gutenberg_id, author, title, filename in GUTENBERG_TEXTS:
        print(f"[{gutenberg_id}] {author} -- {title}")
        if download_text(gutenberg_id, filename, output_dir):
            successes += 1
        else:
            failures += 1
    
    print(f"\nDone. {successes} succeeded, {failures} failed.")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()