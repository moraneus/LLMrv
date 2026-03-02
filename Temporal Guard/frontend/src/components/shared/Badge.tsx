interface BadgeProps {
  variant: "success" | "warning" | "error" | "info" | "neutral";
  children: React.ReactNode;
}

const variantClasses: Record<BadgeProps["variant"], string> = {
  success: "bg-emerald-100 text-emerald-700",
  warning: "bg-amber-100 text-amber-700",
  error: "bg-red-100 text-red-700",
  info: "bg-blue-100 text-blue-700",
  neutral: "bg-slate-100 text-slate-600",
};

export default function Badge({ variant, children }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${variantClasses[variant]}`}
      data-testid="badge"
    >
      {children}
    </span>
  );
}
