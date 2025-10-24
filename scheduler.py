from playwright.sync_api import sync_playwright
from scrapers import DRFScraper, TwinSpiresScraper
import json, time, argparse
from sampler import jsonl_to_dicts, snapshot_to_pools, run_analysis
from random import randrange
from math import exp


def write_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_all_once(site: str, base_url: str, out: str, headless: bool):
    """Run the scraper once for the given site (drf, twinspires) and base (where all races are listed) URL, writing output to the specified file."""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        if site == "drf":
            scraper = DRFScraper()
        elif site == "twinspires":
            scraper = TwinSpiresScraper()
        else:
            raise ValueError("unknown site")

        # Scrape ALL races on that card
        snaps = scraper.scrape_all_races(page, base_url)
        for s in snaps:
            write_jsonl(out, s)
            print(f"Wrote {s['source']} {s['track']} R{s['race_number']} to {out}")

        browser.close()

def scrape_and_analyze(site: str, base_url: str, out: str, headless: bool, print_data: bool = True):
    snapshot = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        
        page = browser.new_page()

        if site == "drf":
            scraper = DRFScraper()
        elif site == "twinspires":
            scraper = TwinSpiresScraper()
        else:
            raise ValueError("unknown site")
        
        snapshot = scraper.scrape_race(page, base_url)
        write_jsonl(out, snapshot)
        print(f"Wrote {snapshot['track']} R{snapshot['race_number']} to {out}")
        browser.close()

    win_pool, show_pool = snapshot_to_pools(snapshot)
    print(f"Track: {snapshot['track']}, Race Number: {snapshot['race_number']}")
    return run_analysis(win_pool, show_pool, print_data)

def get_results(site: str, base_url: str, headless: bool):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        
        page = browser.new_page()

        if site == "drf":
            scraper = DRFScraper()
        elif site == "twinspires":
            scraper = TwinSpiresScraper()
        else:
            raise ValueError("unknown site")
        
        results = scraper.scrape_race_results(page, base_url)
        browser.close()

        return results

def bet_using_kelly(site: str, base_url: str, out: str, headless: bool): # TODO: edit
    base_url = "https://www.twinspires.com/bet/program/classic/mountaineer/mnr/Thoroughbred/"
    balance = 100
    
    for i in range(1,8):
        print()
        url = base_url + str(i) + "/pools"
        bets, expected = scrape_and_analyze(site, url, out, headless, print_data = False)
        
        results = get_results(site, url, headless)
        results["track"] = "mountaineer"
        results["race_number"] = i
        write_jsonl("results.jsonl", results)
        if int(balance*sum(bets.values())) < 1:
            continue
        show_payouts = results["show_payouts"] # gets total return *not* gain

        print(bets)
        print(show_payouts)
        if exp(expected) - 1 < 0.025: # maybe make this a function of the bet size? (if the bet is large, then this should be large)
            print("Skipped!")
            continue
        print("Confidence: ", exp(expected)-1)
        print("Balance before: ", balance)

        for horse, bet in bets.items():
            if horse in show_payouts:
                balance = balance*(1-bet) + balance*bet*show_payouts[horse]/2
            else:
                balance *= (1-bet)
        
        print("Balance after: ", balance)
        print()
        
        if balance <= 0:
            break
    
    print(balance)


if __name__ == "__main__":
    BASE = "https://www.twinspires.com/bet/program/classic/finger-lakes/fl/Thoroughbred/8/pools"
    #BASE = "https://www.twinspires.com/bet/program/classic/parx-racing/prx/Thoroughbred/9/pools"
    BASE = "https://www.twinspires.com/bet/program/classic/horseshoe-indianapolis/ind/Thoroughbred/6/pools"
    #BASE = "https://www.twinspires.com/bet/program/classic/presque-isle/pid/Thoroughbred/1/advanced"
    BASE = "https://www.twinspires.com/bet/program/classic/mountaineer/mnr/Thoroughbred/8/pools"

    # Twinspires requires headless=False to work properly

    balance = 100
    
    bets, expected = scrape_and_analyze("twinspires", BASE, "live_odds_snapshots.jsonl", headless=False)
    print("")
    print("Bets to make:") 
    print("")
    print(f"Expected growth: {exp(expected) - 1}")
    for horse, bet in bets.items():
        if int(bet*balance) >= 1:
            print(f"{horse}: {int(bet*balance)}")
    
    print("")
    print(f"Current balance: {balance}")

    #bet_using_kelly("twinspires", BASE, "live_odds_snapshots.jsonl", headless = False)


    #results = get_results("twinspires", BASE, headless=False)
    #print(results)