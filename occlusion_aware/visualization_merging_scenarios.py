"""
Visualization of Merging Patterns and Occlusion Scenarios
==========================================================
Creates publication-quality figures illustrating:
1. Eight merging patterns (A-H)
2. Truck-car occlusion scenarios
3. Risk metric distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec


def draw_vehicle(ax, x, y, length, width, heading=0, color='blue', 
                 label=None, is_ego=False, alpha=0.8):
    """Draw a vehicle rectangle with rotation."""
    # Create rectangle corners
    corners = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2]
    ])
    
    # Rotate
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ R.T + np.array([x, y])
    
    # Draw
    rect = Polygon(corners, closed=True, facecolor=color, 
                   edgecolor='white' if not is_ego else 'yellow',
                   linewidth=2 if is_ego else 1, alpha=alpha, zorder=4)
    ax.add_patch(rect)
    
    if label:
        ax.text(x, y + width/2 + 0.8, label, ha='center', va='bottom',
               fontsize=8, fontweight='bold' if is_ego else 'normal',
               color='black', bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='white', alpha=0.8))
    
    # Draw heading indicator
    arrow_len = length * 0.3
    ax.arrow(x, y, arrow_len * cos_h, arrow_len * sin_h,
            head_width=0.3, head_length=0.2, fc='white', ec='white', zorder=5)


def create_merging_patterns_figure():
    """Create figure showing all 8 merging patterns."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.25)
    
    patterns = [
        ('A', 'No Vehicles', 'low', {'lv': False, 'rv': False, 'alongside': False}),
        ('B', 'Lead Only', 'low', {'lv': True, 'rv': False, 'alongside': False}),
        ('C', 'Rear→Lead', 'high', {'lv': True, 'rv': False, 'alongside': True, 'transition': 'alongside→lead'}),
        ('D', 'Rear Only', 'medium', {'lv': False, 'rv': True, 'alongside': False}),
        ('E', 'Lead & Rear', 'medium', {'lv': True, 'rv': True, 'alongside': False}),
        ('F', 'Rear→Lead + RV', 'high', {'lv': True, 'rv': True, 'alongside': True, 'transition': 'alongside→lead'}),
        ('G', 'Lead→Rear', 'high', {'lv': False, 'rv': True, 'alongside': True, 'transition': 'lead→rear'}),
        ('H', 'Lead→Rear + LV', 'very high', {'lv': True, 'rv': True, 'alongside': True, 'transition': 'lead→rear'}),
    ]
    
    risk_colors = {'low': '#2ECC71', 'medium': '#F39C12', 'high': '#E74C3C', 'very high': '#8E44AD'}
    
    for i, (code, name, risk, config) in enumerate(patterns):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-5, 25)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        
        # Draw road
        ax.axhspan(-3.5, 0, color='#2C3E50', alpha=0.5)  # Main lane
        ax.axhspan(0, 3.5, color='#34495E', alpha=0.5)   # Accel lane
        ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(-3.5, color='white', linewidth=2)
        ax.axhline(3.5, color='white', linewidth=2)
        
        # Draw merging vehicle (ego)
        draw_vehicle(ax, 10, 1.75, 4.5, 2, color='#2ECC71', label='MV', is_ego=True)
        
        # Draw surrounding vehicles based on config
        if config.get('lv'):
            draw_vehicle(ax, 18, -1.75, 4.5, 2, color='#3498DB', label='LV')
        
        if config.get('rv'):
            draw_vehicle(ax, 3, -1.75, 4.5, 2, color='#9B59B6', label='RV')
        
        if config.get('alongside'):
            # Draw alongside vehicle (gray, will become LV or RV)
            alongside_x = 10 if config.get('transition') == 'alongside→lead' else 10
            alongside_color = '#95A5A6'
            draw_vehicle(ax, alongside_x, -1.75, 4.5, 2, color=alongside_color, 
                        label='Alongside', alpha=0.5)
            
            # Draw arrow showing transition
            if config.get('transition') == 'alongside→lead':
                ax.annotate('', xy=(16, -1.75), xytext=(12, -1.75),
                           arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2))
            elif config.get('transition') == 'lead→rear':
                ax.annotate('', xy=(4, -1.75), xytext=(12, -1.75),
                           arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))
        
        # Draw merge arrow
        ax.annotate('', xy=(12, -0.5), xytext=(10, 1.5),
                   arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))
        
        # Title with risk level
        risk_color = risk_colors[risk]
        ax.set_title(f'Pattern {code}: {name}\nRisk: {risk.upper()}', 
                    fontsize=10, fontweight='bold', color=risk_color, pad=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#4A4A6A')
    
    fig.suptitle('Eight Merging Patterns (Wang et al. 2024)', 
                fontsize=14, fontweight='bold', color='white', y=0.98)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ECC71', label='Merging Vehicle (MV)'),
        mpatches.Patch(facecolor='#3498DB', label='Lead Vehicle (LV)'),
        mpatches.Patch(facecolor='#9B59B6', label='Rear Vehicle (RV)'),
        mpatches.Patch(facecolor='#95A5A6', label='Alongside (transitioning)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              facecolor='#1a1a2e', edgecolor='#4A4A6A', labelcolor='white',
              bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def create_occlusion_scenarios_figure():
    """Create figure showing occlusion scenarios for PINN uncertainty modeling."""
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(1, 2, wspace=0.3)
    
    # Scenario 1: Front Occlusion
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0d1117')
    ax1.set_xlim(-5, 40)
    ax1.set_ylim(-8, 8)
    ax1.set_aspect('equal')
    
    # Draw road
    ax1.axhspan(-3.5, 0, color='#2C3E50', alpha=0.5)
    ax1.axhspan(0, 3.5, color='#34495E', alpha=0.5)
    ax1.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Vehicles
    draw_vehicle(ax1, 30, -1.75, 4.5, 2, color='#3498DB', label='Car A (ahead)')
    draw_vehicle(ax1, 18, -1.75, 12, 2.5, color='#E74C3C', label='TRUCK (ego)', is_ego=True)
    draw_vehicle(ax1, 5, -1.75, 4.5, 2, color='#9B59B6', label='Car B (following)')
    
    # Occlusion zone (shadow)
    shadow = Polygon([[24, -3.5], [24, 0], [40, 5], [40, -8]], 
                     closed=True, facecolor='gray', alpha=0.3, zorder=1)
    ax1.add_patch(shadow)
    
    # Sight lines
    ax1.plot([5, 24], [-1.75, -3.5], 'r--', linewidth=1.5, alpha=0.7)
    ax1.plot([5, 24], [-1.75, 0], 'r--', linewidth=1.5, alpha=0.7)
    
    # "Can't see" label
    ax1.text(32, 3, "Can't see\nCar A", ha='center', va='center',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='red'))
    
    ax1.set_title('Scenario 1: Front Occlusion\n(Car behind truck can\'t see ahead)', 
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax1.set_xlabel('Car B has uncertainty about traffic ahead', fontsize=10, color='#BDC3C7')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_color('#4A4A6A')
    
    # Scenario 2: Lateral Occlusion (Merge Conflict)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0d1117')
    ax2.set_xlim(-5, 35)
    ax2.set_ylim(-12, 8)
    ax2.set_aspect('equal')
    
    # Draw road with merge
    ax2.axhspan(-7, -3.5, color='#2C3E50', alpha=0.5)   # Left lane
    ax2.axhspan(-3.5, 0, color='#34495E', alpha=0.5)    # Middle (truck lane)
    ax2.axhspan(0, 4, color='#4A5568', alpha=0.5)       # Accel lane
    ax2.axhline(-3.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
    
    # Vehicles
    draw_vehicle(ax2, 15, -5.25, 4.5, 2, color='#3498DB', label='Car C (left)')
    draw_vehicle(ax2, 15, -1.75, 12, 2.5, color='#E74C3C', label='TRUCK (ego)', is_ego=True)
    draw_vehicle(ax2, 15, 2, 4.5, 2, color='#2ECC71', label='Car D (merging)')
    
    # Occlusion zone
    shadow2 = Polygon([[15, 0], [15, -3.5], [35, -10], [35, 6]], 
                      closed=True, facecolor='gray', alpha=0.3, zorder=1)
    ax2.add_patch(shadow2)
    
    # Sight lines from Car C
    ax2.plot([15, 21], [-5.25, 0], 'r--', linewidth=1.5, alpha=0.7)
    ax2.plot([15, 9], [-5.25, 0], 'r--', linewidth=1.5, alpha=0.7)
    
    # Merge arrow for Car D
    ax2.annotate('', xy=(17, -1), xytext=(15, 1.5),
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2))
    
    # Conflict zone
    conflict_zone = Circle((17, -1), 2, facecolor='none', 
                           edgecolor='yellow', linewidth=2, linestyle='--', zorder=6)
    ax2.add_patch(conflict_zone)
    ax2.text(17, -1, '⚠', ha='center', va='center', fontsize=16, color='yellow', zorder=7)
    
    # Labels
    ax2.text(25, 4, "Car C can't see\nCar D merging", ha='center', va='center',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='red'))
    
    ax2.text(17, -4, 'Potential\nConflict', ha='center', va='center',
            fontsize=9, color='yellow', fontweight='bold')
    
    ax2.set_title('Scenario 2: Lateral Occlusion\n(Left car can\'t see merging car on right)', 
                 fontsize=12, fontweight='bold', color='white', pad=10)
    ax2.set_xlabel('Both vehicles have incomplete information → conflict risk', 
                  fontsize=10, color='#BDC3C7')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_color('#4A4A6A')
    
    fig.suptitle('Occlusion Scenarios for PINN Uncertainty Modeling', 
                fontsize=14, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def create_six_vehicle_context_figure():
    """Create figure showing six-vehicle context around ego truck."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-20, 40)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    
    # Draw lanes
    lane_width = 3.5
    for i in range(-2, 3):
        y = i * lane_width
        ax.axhline(y, color='white', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(-7, color='white', linewidth=2)  # Road edges
    ax.axhline(10.5, color='white', linewidth=2)
    
    # Labels for lanes
    ax.text(-18, -5.25, 'Main Lane 1', fontsize=9, color='#BDC3C7')
    ax.text(-18, -1.75, 'Main Lane 2', fontsize=9, color='#BDC3C7')
    ax.text(-18, 1.75, 'Main Lane 3', fontsize=9, color='#BDC3C7')
    ax.text(-18, 5.25, 'Accel Lane', fontsize=9, color='#BDC3C7')
    
    # Ego truck (center)
    draw_vehicle(ax, 10, 0, 12, 2.5, color='#E74C3C', label='EGO TRUCK', is_ego=True)
    
    # Six surrounding vehicles
    # Left side
    draw_vehicle(ax, 25, -3.5, 4.5, 2, color='#3498DB', label='left_lead')
    draw_vehicle(ax, 10, -3.5, 4.5, 2, color='#3498DB', label='left_alongside')
    draw_vehicle(ax, -5, -3.5, 4.5, 2, color='#3498DB', label='left_rear')
    
    # Right side (merging area)
    draw_vehicle(ax, 28, 3.5, 4.5, 2, color='#2ECC71', label='right_lead')
    draw_vehicle(ax, 12, 3.5, 4.5, 2, color='#2ECC71', label='right_alongside\n(MERGING)')
    draw_vehicle(ax, -3, 3.5, 4.5, 2, color='#2ECC71', label='right_rear')
    
    # Same lane
    draw_vehicle(ax, 30, 0, 4.5, 2, color='#F39C12', label='front')
    draw_vehicle(ax, -8, 0, 4.5, 2, color='#F39C12', label='rear')
    
    # Draw observation zone
    obs_rect = plt.Rectangle((-15, -10), 55, 20, fill=False, 
                             edgecolor='cyan', linewidth=2, linestyle=':', zorder=1)
    ax.add_patch(obs_rect)
    ax.text(25, 8.5, 'Observation Zone', fontsize=10, color='cyan')
    
    # Occlusion shadows
    # From left_rear's perspective (can't see right_alongside due to truck)
    shadow = Polygon([[10, 3], [4, -1.25], [40, -10], [40, 10]], 
                     closed=True, facecolor='gray', alpha=0.2, zorder=1)
    ax.add_patch(shadow)
    
    # Add conflict annotation
    ax.annotate('Potential\nConflict', xy=(12, 0.5), xytext=(20, -6),
               fontsize=9, color='yellow', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
    
    ax.set_title('Six-Vehicle Surrounding Context for Ego Truck\n'
                '(For PINN interaction field learning)',
                fontsize=12, fontweight='bold', color='white', pad=15)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#4A4A6A')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E74C3C', label='Ego Truck'),
        mpatches.Patch(facecolor='#3498DB', label='Left Side Vehicles'),
        mpatches.Patch(facecolor='#2ECC71', label='Right Side (Merging)'),
        mpatches.Patch(facecolor='#F39C12', label='Same Lane'),
        mpatches.Patch(facecolor='gray', alpha=0.3, label='Occlusion Zone'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
             facecolor='#1a1a2e', edgecolor='#4A4A6A', labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_risk_comparison_figure():
    """Create figure comparing risk levels across patterns."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # Simulated data (based on paper findings)
    np.random.seed(42)
    
    # TTC values by pattern (lower is more risky)
    ttc_means = [8.5, 6.2, 2.1, 4.5, 3.8, 1.8, 1.5, 1.2]
    ttc_stds = [2.0, 1.5, 0.8, 1.2, 1.0, 0.6, 0.5, 0.4]
    
    # Merging duration by pattern
    duration_means = [3.5, 4.2, 5.8, 4.0, 4.5, 6.2, 5.5, 6.8]
    duration_stds = [0.8, 1.0, 1.5, 0.9, 1.1, 1.8, 1.2, 1.5]
    
    # High risk ratio by pattern
    high_risk_ratio = [0.05, 0.10, 0.45, 0.20, 0.25, 0.55, 0.60, 0.72]
    
    colors = ['#2ECC71', '#2ECC71', '#E74C3C', '#F39C12', '#F39C12', 
              '#E74C3C', '#E74C3C', '#8E44AD']
    
    # Plot 1: TTC by pattern
    ax1 = axes[0]
    ax1.set_facecolor('#0d1117')
    bars = ax1.bar(patterns, ttc_means, yerr=ttc_stds, capsize=3, color=colors, alpha=0.8)
    ax1.axhline(2.5, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax1.set_xlabel('Merging Pattern', fontsize=10, color='white')
    ax1.set_ylabel('TTC at Merge (seconds)', fontsize=10, color='white')
    ax1.set_title('Time-to-Collision by Pattern', fontsize=11, fontweight='bold', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#4A4A6A', labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_color('#4A4A6A')
    
    # Plot 2: Duration by pattern
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    ax2.bar(patterns, duration_means, yerr=duration_stds, capsize=3, color=colors, alpha=0.8)
    ax2.set_xlabel('Merging Pattern', fontsize=10, color='white')
    ax2.set_ylabel('Merging Duration (seconds)', fontsize=10, color='white')
    ax2.set_title('Merging Duration by Pattern', fontsize=11, fontweight='bold', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#4A4A6A')
    
    # Plot 3: High risk ratio
    ax3 = axes[2]
    ax3.set_facecolor('#0d1117')
    bars3 = ax3.bar(patterns, [r * 100 for r in high_risk_ratio], color=colors, alpha=0.8)
    ax3.set_xlabel('Merging Pattern', fontsize=10, color='white')
    ax3.set_ylabel('High Risk Events (%)', fontsize=10, color='white')
    ax3.set_title('High Risk Proportion by Pattern', fontsize=11, fontweight='bold', color='white')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#4A4A6A')
    
    # Add pattern descriptions
    fig.text(0.5, 0.02, 
            'Low Risk: A(free), B(lead only) | Medium Risk: D(rear only), E(lead+rear) | '
            'High Risk: C,F(alongside→lead), G,H(cut-in)',
            ha='center', fontsize=9, color='#BDC3C7')
    
    fig.suptitle('Risk Metrics Across Merging Patterns', 
                fontsize=13, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


if __name__ == '__main__':
    # Generate all figures
    output_dir = '/mnt/user-data/outputs'
    
    print("Creating merging patterns figure...")
    fig1 = create_merging_patterns_figure()
    fig1.savefig(f'{output_dir}/merging_patterns.png', dpi=150, 
                bbox_inches='tight', facecolor=fig1.get_facecolor())
    plt.close(fig1)
    
    print("Creating occlusion scenarios figure...")
    fig2 = create_occlusion_scenarios_figure()
    fig2.savefig(f'{output_dir}/occlusion_scenarios.png', dpi=150,
                bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close(fig2)
    
    print("Creating six-vehicle context figure...")
    fig3 = create_six_vehicle_context_figure()
    fig3.savefig(f'{output_dir}/six_vehicle_context.png', dpi=150,
                bbox_inches='tight', facecolor=fig3.get_facecolor())
    plt.close(fig3)
    
    print("Creating risk comparison figure...")
    fig4 = create_risk_comparison_figure()
    fig4.savefig(f'{output_dir}/risk_comparison.png', dpi=150,
                bbox_inches='tight', facecolor=fig4.get_facecolor())
    plt.close(fig4)
    
    print("All figures saved successfully!")
